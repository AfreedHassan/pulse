use std::ffi::c_void;
use std::sync::mpsc;
use std::thread;

#[allow(non_upper_case_globals)]
mod ffi {
    use std::ffi::c_void;

    pub type CGEventRef = *mut c_void;
    pub type CGEventFlags = u64;
    pub type CGEventTapProxy = *mut c_void;
    pub type CFMachPortRef = *mut c_void;
    pub type CFRunLoopSourceRef = *mut c_void;
    pub type CFRunLoopRef = *mut c_void;
    pub type CFAllocatorRef = *const c_void;
    pub type CFStringRef = *const c_void;
    pub type CFIndex = i64;

    pub const kCGSessionEventTap: u32 = 1;
    pub const kCGHeadInsertEventTap: u32 = 0;
    pub const kCGEventTapOptionListenOnly: u32 = 1;

    /// Event type for modifier key changes (Fn, Shift, Cmd, etc.).
    pub const kCGEventFlagsChanged: u32 = 12;

    /// Fn/Globe key modifier mask.
    pub const kCGEventFlagMaskSecondaryFn: CGEventFlags = 1 << 23;

    /// Event mask: listen for flagsChanged events.
    pub const CGEventMaskBitFlagsChanged: u64 = 1 << 12;

    pub type CGEventTapCallBack = unsafe extern "C" fn(
        proxy: CGEventTapProxy,
        event_type: u32,
        event: CGEventRef,
        user_info: *mut c_void,
    ) -> CGEventRef;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        pub fn CGEventTapCreate(
            tap: u32,
            place: u32,
            options: u32,
            events_of_interest: u64,
            callback: CGEventTapCallBack,
            user_info: *mut c_void,
        ) -> CFMachPortRef;
        pub fn CGEventGetFlags(event: CGEventRef) -> CGEventFlags;
        pub fn CGEventTapEnable(tap: CFMachPortRef, enable: bool);
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFMachPortCreateRunLoopSource(
            allocator: CFAllocatorRef,
            port: CFMachPortRef,
            order: CFIndex,
        ) -> CFRunLoopSourceRef;
        pub fn CFRunLoopGetCurrent() -> CFRunLoopRef;
        pub fn CFRunLoopAddSource(rl: CFRunLoopRef, source: CFRunLoopSourceRef, mode: CFStringRef);
        pub fn CFRunLoopRun();
        pub fn CFRelease(cf: *const c_void);
        pub static kCFRunLoopCommonModes: CFStringRef;
    }
}

/// State passed to the event tap callback.
struct CallbackState {
    sender: mpsc::Sender<()>,
    fn_held: bool,
}

unsafe extern "C" fn hotkey_callback(
    _proxy: ffi::CGEventTapProxy,
    event_type: u32,
    event: ffi::CGEventRef,
    user_info: *mut c_void,
) -> ffi::CGEventRef {
    if event_type == ffi::kCGEventFlagsChanged {
        unsafe {
            let state = &mut *(user_info as *mut CallbackState);
            let flags = ffi::CGEventGetFlags(event);
            let fn_down = (flags & ffi::kCGEventFlagMaskSecondaryFn) != 0;

            if fn_down && !state.fn_held {
                // Fn just pressed — signal start.
                state.fn_held = true;
                let _ = state.sender.send(());
            } else if !fn_down && state.fn_held {
                // Fn just released — signal stop.
                state.fn_held = false;
                let _ = state.sender.send(());
            }
        }
    }
    event
}

/// Spawn a background thread that listens for the Fn (Globe 🌐) key globally.
/// Returns a receiver that fires on Fn press (start) and Fn release (stop).
///
/// Requires:
/// - Accessibility permissions (System Settings → Privacy & Security → Accessibility)
/// - "Press 🌐 key to: Do Nothing" (System Settings → Keyboard) so macOS doesn't intercept it
pub fn listen() -> mpsc::Receiver<()> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || unsafe {
        let state_ptr = Box::into_raw(Box::new(CallbackState {
            sender: tx,
            fn_held: false,
        }));

        let tap = ffi::CGEventTapCreate(
            ffi::kCGSessionEventTap,
            ffi::kCGHeadInsertEventTap,
            ffi::kCGEventTapOptionListenOnly,
            ffi::CGEventMaskBitFlagsChanged,
            hotkey_callback,
            state_ptr as *mut c_void,
        );

        if tap.is_null() {
            eprintln!(
                "Failed to create event tap. \
                 Grant Accessibility permissions: System Settings → Privacy & Security → Accessibility."
            );
            return;
        }

        let source = ffi::CFMachPortCreateRunLoopSource(std::ptr::null(), tap, 0);
        let run_loop = ffi::CFRunLoopGetCurrent();
        ffi::CFRunLoopAddSource(run_loop, source, ffi::kCFRunLoopCommonModes);
        ffi::CGEventTapEnable(tap, true);

        eprintln!("Global hotkey active: hold Fn (🌐) to record");

        // Blocks forever, processing hotkey events.
        ffi::CFRunLoopRun();

        // Cleanup (unreachable in practice).
        let _ = Box::from_raw(state_ptr);
        ffi::CFRelease(source);
        ffi::CFRelease(tap);
    });

    rx
}
