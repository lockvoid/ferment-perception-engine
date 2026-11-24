//! Safe, blocking Windows ML API. See README for ownership and worker contract.
#![cfg(windows)]

use std::{
    collections::BTreeMap,
    ffi::{CStr, OsString, c_char, c_void},
    os::windows::ffi::OsStringExt,
    path::PathBuf,
    ptr,
    sync::{Mutex, OnceLock, PoisonError},
};

use libloading::Library;
use ort::{AsPointer, environment::Environment, session::Session};

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
mod native_ort {
    include!(concat!(env!("OUT_DIR"), "/ort.rs"));
}

type Result<T> = std::result::Result<T, String>;
type Handle = *mut c_void;
type HResult = i32;

// ABI from the pinned NuGet's WinMLEpCatalog.h: UTF-8 pointers, C enums (i32),
// size_t (usize), HRESULT/BOOL (i32), and WINAPI/CALLBACK (system convention).
#[repr(C)]
struct EpInfo {
    name: *const c_char,
    version: *const c_char,
    package_family_name: *const c_char,
    library_path: *const c_char,
    package_root_path: *const c_char,
    ready_state: i32,
    certification: i32,
}

type Callback = unsafe extern "system" fn(Handle, *const EpInfo, *mut c_void) -> i32;

struct Api {
    _library: Library,
    create: unsafe extern "system" fn(*mut Handle) -> HResult,
    release: unsafe extern "system" fn(Handle),
    enumerate: unsafe extern "system" fn(Handle, Callback, *mut c_void) -> HResult,
    ensure_ready: unsafe extern "system" fn(Handle) -> HResult,
    path_size: unsafe extern "system" fn(Handle, *mut usize) -> HResult,
    path: unsafe extern "system" fn(Handle, usize, *mut c_char, *mut usize) -> HResult,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleExW(flags: u32, address: *const c_void, module: *mut Handle) -> i32;
    fn GetModuleFileNameW(module: Handle, path: *mut u16, size: u32) -> u32;
}

fn runtime_path() -> Result<PathBuf> {
    let mut module = ptr::null_mut();
    // FROM_ADDRESS | UNCHANGED_REFCOUNT. ORT owns its loaded DLL for the process
    // lifetime, so this borrowed handle stays live while we copy its path.
    let ok = unsafe {
        GetModuleHandleExW(
            0x4 | 0x2,
            ort::api().CreateSession as *const c_void,
            &mut module,
        )
    };
    if ok == 0 {
        return Err(format!(
            "locate loaded ORT: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut path = vec![0u16; 32768];
    let len = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
    if len == 0 || len >= path.len() {
        return Err("cannot resolve the loaded ORT module path".into());
    }
    Ok(PathBuf::from(OsString::from_wide(&path[..len])))
}

impl Api {
    fn load() -> Result<Self> {
        let path = runtime_path()?.with_file_name("Microsoft.Windows.AI.MachineLearning.dll");
        // Trusted application runtime sibling, never a PATH search. Function
        // signatures below are the matching public WinMLEpCatalog.h ABI.
        unsafe {
            let library = Library::new(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(Self {
                create: *library
                    .get(b"WinMLEpCatalogCreate\0")
                    .map_err(|e| e.to_string())?,
                release: *library
                    .get(b"WinMLEpCatalogRelease\0")
                    .map_err(|e| e.to_string())?,
                enumerate: *library
                    .get(b"WinMLEpCatalogEnumProviders\0")
                    .map_err(|e| e.to_string())?,
                ensure_ready: *library
                    .get(b"WinMLEpEnsureReady\0")
                    .map_err(|e| e.to_string())?,
                path_size: *library
                    .get(b"WinMLEpGetLibraryPathSize\0")
                    .map_err(|e| e.to_string())?,
                path: *library
                    .get(b"WinMLEpGetLibraryPath\0")
                    .map_err(|e| e.to_string())?,
                _library: library,
            })
        }
    }
}

struct Catalog<'a> {
    handle: Handle,
    api: &'a Api,
}

impl Drop for Catalog<'_> {
    fn drop(&mut self) {
        // Exactly the handle created by this API; all EP uses have completed.
        unsafe { (self.api.release)(self.handle) };
    }
}

fn hresult(operation: &str, hr: HResult) -> Result<()> {
    if hr < 0 {
        Err(format!("{operation}: HRESULT 0x{:08X}", hr as u32))
    } else {
        Ok(())
    }
}

struct Entry {
    handle: Handle,
    name: String,
}

unsafe extern "system" fn collect(ep: Handle, info: *const EpInfo, context: *mut c_void) -> i32 {
    if info.is_null() || ep.is_null() || context.is_null() {
        return 1;
    }
    // EnumProviders calls synchronously, with valid borrowed pointers. The
    // Vec lives until enumeration returns. Only certified providers are used.
    let info = unsafe { &*info };
    if info.certification == 1 && !info.name.is_null() {
        let entries = unsafe { &mut *context.cast::<Vec<Entry>>() };
        entries.push(Entry {
            handle: ep,
            name: unsafe { CStr::from_ptr(info.name) }
                .to_string_lossy()
                .into_owned(),
        });
    }
    1
}

#[derive(Clone, Debug)]
pub struct Provider {
    pub name: String,
    pub library: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Preparation {
    pub providers: Vec<Provider>,
    pub devices: Vec<String>,
    pub warnings: Vec<String>,
}

struct Registration {
    provider: Provider,
    _handle: ort::ep::ExecutionProviderLibrary,
}

static API: OnceLock<Result<Api>> = OnceLock::new();
static REGISTERED: Mutex<BTreeMap<String, Registration>> = Mutex::new(BTreeMap::new());

/// Discover, prepare (possibly download), and register certified EPs into the
/// same ORT environment sessions use. Call only on a blocking worker.
pub fn prepare() -> Result<Preparation> {
    let environment = Environment::current().map_err(|e| e.to_string())?;
    let api = API.get_or_init(Api::load).as_ref().map_err(Clone::clone)?;
    let mut registered = REGISTERED.lock().unwrap_or_else(PoisonError::into_inner);
    let mut handle = ptr::null_mut();
    hresult("WinMLEpCatalogCreate", unsafe { (api.create)(&mut handle) })?;
    if handle.is_null() {
        return Err("Windows ML returned a null catalog".into());
    }
    let catalog = Catalog { handle, api };
    let mut entries = Vec::<Entry>::new();
    hresult("WinMLEpCatalogEnumProviders", unsafe {
        (api.enumerate)(
            catalog.handle,
            collect,
            (&mut entries as *mut Vec<Entry>).cast(),
        )
    })?;
    let mut warnings = Vec::new();
    for entry in entries {
        if registered.contains_key(&entry.name) {
            continue;
        }
        let register = || -> Result<Registration> {
            // Required even when installed: EnsureReady also adds package
            // dependencies to this process. The catalog owns the EP handle.
            hresult("WinMLEpEnsureReady", unsafe {
                (api.ensure_ready)(entry.handle)
            })?;
            let mut size = 0;
            hresult("WinMLEpGetLibraryPathSize", unsafe {
                (api.path_size)(entry.handle, &mut size)
            })?;
            if size == 0 || size > 65536 {
                return Err(format!("invalid EP library path size: {size}"));
            }
            let mut bytes = vec![0u8; size];
            hresult("WinMLEpGetLibraryPath", unsafe {
                (api.path)(
                    entry.handle,
                    size,
                    bytes.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
            })?;
            let path = CStr::from_bytes_until_nul(&bytes)
                .map_err(|e| e.to_string())?
                .to_str()
                .map_err(|e| e.to_string())?;
            let library = PathBuf::from(path);
            if !library.is_absolute() || !library.is_file() {
                return Err(format!(
                    "catalog returned an unavailable EP library: {path}"
                ));
            }
            let handle = environment
                .register_ep_library(&entry.name, &library)
                .map_err(|e| e.to_string())?;
            Ok(Registration {
                provider: Provider {
                    name: entry.name.clone(),
                    library,
                },
                _handle: handle,
            })
        };
        match register() {
            Ok(value) => {
                registered.insert(entry.name, value);
            }
            Err(error) => warnings.push(format!("{}: {error}", entry.name)),
        }
    }
    // Microsoft permits releasing the catalog after registration. The DLL
    // and registered EPs remain alive for all sessions in this environment.
    Ok(Preparation {
        providers: registered.values().map(|r| r.provider.clone()).collect(),
        devices: environment
            .devices()
            .map(|d| {
                Ok(format!(
                    "{} / {} / {:?} / {}",
                    d.ep().map_err(|e| e.to_string())?,
                    d.vendor().map_err(|e| e.to_string())?,
                    d.ty(),
                    d.id()
                ))
            })
            .collect::<Result<_>>()?,
        warnings,
    })
}

#[derive(Clone, Debug)]
pub struct AssignedNode {
    pub name: String,
    pub domain: String,
    pub operator: String,
}

#[derive(Clone, Debug)]
pub struct AssignedSubgraph {
    pub provider: String,
    pub nodes: Vec<AssignedNode>,
}

fn status(value: *mut native_ort::OrtStatus) -> Result<()> {
    // All callers pass the just-returned ORT status; conversion consumes it.
    unsafe { ort::Error::result_from_status(ort::sys::OrtStatusPtr(value.cast())) }
        .map_err(|e| e.to_string())
}

struct NativeOrt {
    _library: Library,
    api: &'static native_ort::OrtApi,
}

static NATIVE_ORT: OnceLock<Result<NativeOrt>> = OnceLock::new();

fn assignment_api() -> Result<&'static native_ort::OrtApi> {
    let native = NATIVE_ORT
        .get_or_init(|| unsafe {
            // The same DLL ort already loaded. Get API 28 using its own header:
            // ort-sys rc.12 omits KernelInfo_GetOperatorSinceVersion, shifting all
            // later members by one. Its late API-24 table cannot be used here.
            let library = Library::new(runtime_path()?).map_err(|e| e.to_string())?;
            let get_base = library
                .get::<unsafe extern "system" fn() -> *const native_ort::OrtApiBase>(
                    b"OrtGetApiBase\0",
                )
                .map_err(|e| e.to_string())?;
            let base = get_base().as_ref().ok_or("ORT returned a null API base")?;
            let version = copy_string(base.GetVersionString.ok_or("ORT has no version API")?())?;
            let minor = version
                .split('.')
                .nth(1)
                .and_then(|s| s.parse::<u32>().ok());
            if !version.starts_with("1.") || minor.is_none_or(|v| v < native_ort::ORT_API_VERSION) {
                return Err(format!(
                    "assignment needs Windows ML ORT 1.28+, loaded {version}"
                ));
            }
            let api = base.GetApi.ok_or("ORT has no GetApi")?(native_ort::ORT_API_VERSION)
                .as_ref()
                .ok_or("Windows ML assignment requires ORT API 28")?;
            // OnceLock retains the DLL for the entire lifetime of this reference.
            Ok(NativeOrt {
                _library: library,
                api,
            })
        })
        .as_ref()
        .map_err(Clone::clone)?;
    Ok(native.api)
}

unsafe fn copy_string(value: *const c_char) -> Result<String> {
    if value.is_null() {
        return Err("ORT returned a null assignment string".into());
    }
    // ORT owns this null-terminated UTF-8 string for the borrowed session.
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(str::to_owned)
        .map_err(|e| e.to_string())
}

/// Copy actual graph assignments, including the original nodes in fused EP
/// subgraphs. The caller must have enabled assignment recording on the session.
pub fn graph_assignment(session: &Session) -> Result<Vec<AssignedSubgraph>> {
    let api = assignment_api()?;
    let mut graphs = ptr::null();
    let mut count = 0;
    // Session borrow keeps all arrays/strings below alive; owned copies escape.
    unsafe {
        status((api
            .Session_GetEpGraphAssignmentInfo
            .ok_or("missing assignment API")?)(
            session.ptr().cast(),
            &mut graphs,
            &mut count,
        ))?;
        if count == 0 {
            return Ok(Vec::new());
        }
        if graphs.is_null() {
            return Err("ORT returned null assignment array".into());
        }
        let mut result = Vec::with_capacity(count);
        for &graph in std::slice::from_raw_parts(graphs, count) {
            let mut name = ptr::null();
            status((api
                .EpAssignedSubgraph_GetEpName
                .ok_or("missing EP name API")?)(
                graph, &mut name
            ))?;
            let provider = copy_string(name)?;
            let mut nodes = ptr::null();
            let mut length = 0;
            status((api
                .EpAssignedSubgraph_GetNodes
                .ok_or("missing EP nodes API")?)(
                graph,
                &mut nodes,
                &mut length,
            ))?;
            let mut owned = Vec::with_capacity(length);
            if length > 0 {
                if nodes.is_null() {
                    return Err("ORT returned null node array".into());
                }
                for &node in std::slice::from_raw_parts(nodes, length) {
                    status((api
                        .EpAssignedNode_GetName
                        .ok_or("missing node name API")?)(
                        node, &mut name
                    ))?;
                    let node_name = copy_string(name)?;
                    status((api
                        .EpAssignedNode_GetDomain
                        .ok_or("missing node domain API")?)(
                        node, &mut name
                    ))?;
                    let domain = copy_string(name)?;
                    status((api
                        .EpAssignedNode_GetOperatorType
                        .ok_or("missing node type API")?)(
                        node, &mut name
                    ))?;
                    owned.push(AssignedNode {
                        name: node_name,
                        domain,
                        operator: copy_string(name)?,
                    });
                }
            }
            result.push(AssignedSubgraph {
                provider,
                nodes: owned,
            });
        }
        Ok(result)
    }
}
