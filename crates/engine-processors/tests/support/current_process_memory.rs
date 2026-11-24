// Process memory for current iOS parity gates. macOS reads the same
// TASK_VM_INFO.phys_footprint field as MemoryProbe.swift (SDK task_info.h rev1).
// Android/Linux have no Mach physical-footprint counter; VmRSS is named explicitly.
#[cfg(target_os = "macos")]
// This test-only OS counter needs the documented Mach FFI; the production crate remains safe.
#[allow(unsafe_code)]
pub fn footprint_bytes() -> u64 {
    #[repr(C)]
    struct TaskVmInfoRev1 {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
    }
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    assert_eq!(std::mem::size_of::<TaskVmInfoRev1>(), 152);
    let mut info = std::mem::MaybeUninit::<TaskVmInfoRev1>::zeroed();
    let mut count = (std::mem::size_of::<TaskVmInfoRev1>() / std::mem::size_of::<i32>()) as u32;
    // SAFETY: the rev1 C-layout prefix contains exactly `count` integer_t cells,
    // the caller owns it, and the current task port is valid for task_info.
    let status = unsafe {
        task_info(
            mach_task_self_,
            22,
            info.as_mut_ptr().cast::<i32>(),
            &mut count,
        )
    };
    assert_eq!(status, 0, "TASK_VM_INFO failed");
    assert!(count >= 38, "kernel omitted the physical-footprint field");
    // SAFETY: a successful task_info filled the requested rev1 prefix above.
    unsafe { info.assume_init().phys_footprint }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn footprint_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 1024
}
