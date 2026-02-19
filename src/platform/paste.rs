use std::io::Write;
use std::process::{Command, Stdio};

/// macOS virtual keycode for 'v'.
const KEYCODE_V: u16 = 9;

// CoreGraphics FFI — just the handful of functions we need for Cmd+V.
#[allow(non_upper_case_globals)]
mod cg {
    use std::ffi::c_void;

    pub type CGEventRef = *mut c_void;
    pub type CGEventSourceRef = *mut c_void;
    pub type CGKeyCode = u16;
    pub type CGEventFlags = u64;
    pub type CGEventTapLocation = u32;
    pub type CGEventSourceStateID = u32;

    /// HID system state — most reliable for synthetic input.
    pub const kCGEventSourceStateHIDSystemState: CGEventSourceStateID = 1;
    /// Post at the HID level so it reaches all apps.
    pub const kCGHIDEventTap: CGEventTapLocation = 0;
    /// Cmd key mask.
    pub const kCGEventFlagMaskCommand: CGEventFlags = 1 << 20;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        pub fn CGEventSourceCreate(stateID: CGEventSourceStateID) -> CGEventSourceRef;
        pub fn CGEventCreateKeyboardEvent(
            source: CGEventSourceRef,
            virtualKey: CGKeyCode,
            keyDown: bool,
        ) -> CGEventRef;
        pub fn CGEventSetFlags(event: CGEventRef, flags: CGEventFlags);
        pub fn CGEventPost(tap: CGEventTapLocation, event: CGEventRef);
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFRelease(cf: *const c_void);
    }
}

/// Copy `text` into the macOS system clipboard via `pbcopy`.
pub fn clipboard_set(text: &str) -> anyhow::Result<()> {
    let mut child = Command::new("pbcopy").stdin(Stdio::piped()).spawn()?;
    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    let status = child.wait()?;
    anyhow::ensure!(status.success(), "pbcopy exited with {}", status);
    Ok(())
}

/// Simulate Cmd+V in the frontmost application using CoreGraphics events.
///
/// Requires Accessibility permissions (System Settings → Privacy & Security → Accessibility).
fn paste_to_frontmost() -> anyhow::Result<()> {
    unsafe {
        let source = cg::CGEventSourceCreate(cg::kCGEventSourceStateHIDSystemState);
        anyhow::ensure!(!source.is_null(), "Failed to create CGEventSource");

        let key_down = cg::CGEventCreateKeyboardEvent(source, KEYCODE_V, true);
        anyhow::ensure!(!key_down.is_null(), "Failed to create keyDown event");
        cg::CGEventSetFlags(key_down, cg::kCGEventFlagMaskCommand);

        let key_up = cg::CGEventCreateKeyboardEvent(source, KEYCODE_V, false);
        anyhow::ensure!(!key_up.is_null(), "Failed to create keyUp event");
        cg::CGEventSetFlags(key_up, cg::kCGEventFlagMaskCommand);

        cg::CGEventPost(cg::kCGHIDEventTap, key_down);
        cg::CGEventPost(cg::kCGHIDEventTap, key_up);

        cg::CFRelease(key_up);
        cg::CFRelease(key_down);
        cg::CFRelease(source);
    }
    Ok(())
}

/// Copy `text` to the clipboard and simulate Cmd+V to paste into the frontmost app.
pub fn paste_text(text: &str) -> anyhow::Result<()> {
    clipboard_set(text)?;
    paste_to_frontmost()?;
    Ok(())
}
