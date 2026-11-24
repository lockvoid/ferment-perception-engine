//! Safe facade over the public CoreAI Swift API. Calls block a dedicated native
//! worker. Swift owns every tensor through completion; outputs are copied using
//! their logical strides/interleave layout before runtime storage is released.
#![cfg(target_os = "macos")]
#![deny(unsafe_op_in_unsafe_fn)]

use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    ptr::NonNull,
    sync::Mutex,
};

#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum DType {
    F32 = 1,
    F16 = 2,
    I32 = 3,
    I64 = 4,
}
impl DType {
    fn width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::I64 => 8,
            _ => 4,
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
use ffi::*;

// All returned strings are strdup-owned by the bridge, including errors.
unsafe fn take_string(value: *mut c_char) -> String {
    let text = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe { ferment_coreai_free_string(value) };
    text
}

pub fn device_architecture() -> Option<String> {
    // The Swift function checks availability before touching CoreAI symbols.
    let value = unsafe { ferment_coreai_architecture() };
    (!value.is_null()).then(|| unsafe { take_string(value) })
}

pub struct Model(Mutex<NonNull<c_void>>);

#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    pub expect_frequent_reshapes: bool,
}
// AIModel is Sendable; the retained ModelBox survives until Drop. Rust serializes
// all calls on one instance, including function lookup. Drop needs unique access.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        Self::load_with_options(path, LoadOptions::default())
    }

    pub fn load_with_options(path: &Path, options: LoadOptions) -> Result<Self, String> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| "Model path contains NUL")?;
        let mut error = std::ptr::null_mut();
        // CString and the error slot live through the synchronous Swift call.
        let raw = unsafe {
            ferment_coreai_load(
                path.as_ptr(),
                i32::from(options.expect_frequent_reshapes),
                &mut error,
            )
        };
        if !error.is_null() {
            return Err(unsafe { take_string(error) });
        }
        NonNull::new(raw)
            .map(|raw| Self(Mutex::new(raw)))
            .ok_or_else(|| "CoreAI returned no model".into())
    }

    pub fn predict(&self, function: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        let function = CString::new(function).map_err(|_| "Function contains NUL")?;
        let mut unique = std::collections::HashSet::new();
        for input in inputs {
            if input.shape.len() > 16 || !unique.insert(&input.name) {
                return Err("Invalid input name/rank".into());
            }
            let size = input
                .shape
                .iter()
                .try_fold(input.dtype.width(), |n, d| {
                    usize::try_from(*d)
                        .ok()
                        .filter(|d| *d > 0)
                        .and_then(|d| n.checked_mul(d))
                })
                .ok_or("Invalid tensor shape")?;
            if input.data.len() != size {
                return Err("Input byte length differs from tensor shape".into());
            }
        }
        let names = inputs
            .iter()
            .map(|t| CString::new(t.name.as_str()).map_err(|_| "Input name contains NUL"))
            .collect::<Result<Vec<_>, _>>()?;
        let name_ptrs: Vec<_> = names.iter().map(|s| s.as_ptr()).collect();
        let shapes: Vec<_> = inputs.iter().map(|t| t.shape.as_ptr()).collect();
        let ranks: Vec<_> = inputs.iter().map(|t| t.shape.len() as isize).collect();
        let types: Vec<_> = inputs.iter().map(|t| t.dtype as i32).collect();
        let buffers: Vec<*const c_void> = inputs.iter().map(|t| t.data.as_ptr().cast()).collect();
        let lengths = inputs
            .iter()
            .map(|t| isize::try_from(t.data.len()).map_err(|_| "Input is too large"))
            .collect::<Result<Vec<_>, _>>()?;
        let count = isize::try_from(inputs.len()).map_err(|_| "Too many inputs")?;
        let mut outputs = Outputs::default();
        let model = self
            .0
            .lock()
            .map_err(|_| "CoreAI instance lock was poisoned")?;
        // Every pointer references owned storage above, alive until the bridge
        // returns. The callback is synchronous and does not retain its context.
        let error = unsafe {
            ferment_coreai_predict(
                model.as_ptr(),
                function.as_ptr(),
                count,
                name_ptrs.as_ptr(),
                shapes.as_ptr(),
                ranks.as_ptr(),
                types.as_ptr(),
                buffers.as_ptr(),
                lengths.as_ptr(),
                (&mut outputs as *mut Outputs).cast(),
                Some(receive),
            )
        };
        if !error.is_null() {
            return Err(unsafe { take_string(error) });
        }
        if outputs.failed {
            return Err("Invalid CoreAI output".into());
        }
        Ok(outputs.tensors)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let raw = self.0.get_mut().unwrap_or_else(|e| e.into_inner());
        // Unique owner: no prediction can still hold &self when Drop runs.
        unsafe { ferment_coreai_release(raw.as_ptr()) };
    }
}

#[derive(Default)]
struct Outputs {
    tensors: Vec<Tensor>,
    failed: bool,
}

unsafe extern "C" fn receive(
    context: *mut c_void,
    name: *const c_char,
    shape: *const i64,
    rank: isize,
    dtype: i32,
    bytes: *const c_void,
    length: isize,
) -> i32 {
    // These pointers originate in Bridge.swift's withCString/withUnsafeBytes
    // scopes and are only borrowed here. Copy before returning to Swift.
    if context.is_null() {
        return 1;
    }
    let outputs = unsafe { &mut *context.cast::<Outputs>() };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Option<Tensor> {
        let rank = usize::try_from(rank).ok()?;
        let length = usize::try_from(length).ok()?;
        if name.is_null()
            || rank > 16
            || (rank != 0 && shape.is_null())
            || (length != 0 && bytes.is_null())
        {
            return None;
        }
        let dtype = match dtype {
            1 => DType::F32,
            2 => DType::F16,
            3 => DType::I32,
            4 => DType::I64,
            _ => return None,
        };
        let shape = if rank == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(shape, rank) }.to_vec()
        };
        let expected = shape.iter().try_fold(dtype.width(), |n, d| {
            usize::try_from(*d)
                .ok()
                .filter(|d| *d > 0)
                .and_then(|d| n.checked_mul(d))
        })?;
        if length != expected {
            return None;
        }
        Some(Tensor {
            name: unsafe { CStr::from_ptr(name) }.to_str().ok()?.to_owned(),
            shape,
            dtype,
            data: if length == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), length) }.to_vec()
            },
        })
    }));
    match result {
        Ok(Some(tensor)) if !outputs.tensors.iter().any(|t| t.name == tensor.name) => {
            outputs.tensors.push(tensor);
            0
        }
        _ => {
            outputs.failed = true;
            1
        }
    }
}
