//! Read the focused text field via the macOS Accessibility API.
//!
//! Returns the current text content of the focused UI element (if any),
//! which can be used to give the LLM formatter context about surrounding text.
//! Falls back to `None` on any failure (permission denied, no focused element, etc.).

/// Maximum characters of context to return (tail end of the text field).
const MAX_CONTEXT_CHARS: usize = 500;

#[cfg(target_os = "macos")]
#[allow(non_upper_case_globals)]
mod ax {
    use std::ffi::{c_void, CStr};
    use std::os::raw::c_char;
    use std::ptr;

    pub type AXUIElementRef = *mut c_void;
    pub type AXError = i32;
    pub type CFTypeRef = *const c_void;
    pub type CFStringRef = *const c_void;
    pub type CFIndex = i64;
    pub type CFStringEncoding = u32;

    pub const kCFStringEncodingUTF8: CFStringEncoding = 0x08000100;
    pub const kAXErrorSuccess: AXError = 0;

    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        pub fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        pub fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFRelease(cf: *const c_void);
        pub fn CFStringGetLength(string: CFStringRef) -> CFIndex;
        pub fn CFStringGetCString(
            string: CFStringRef,
            buffer: *mut c_char,
            buffer_size: CFIndex,
            encoding: CFStringEncoding,
        ) -> bool;
        pub fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const c_char,
            encoding: CFStringEncoding,
        ) -> CFStringRef;
    }

    /// Create a CFString from a Rust `&str`. Caller must `CFRelease` the result.
    pub unsafe fn cfstr(s: &str) -> CFStringRef {
        let c = std::ffi::CString::new(s).unwrap();
        unsafe { CFStringCreateWithCString(ptr::null(), c.as_ptr(), kCFStringEncodingUTF8) }
    }

    /// Convert a CFStringRef to a Rust String. Returns `None` if conversion fails.
    pub unsafe fn cfstring_to_string(cf: CFStringRef) -> Option<String> {
        if cf.is_null() {
            return None;
        }
        unsafe {
            let len = CFStringGetLength(cf);
            // UTF-8 can be up to 4 bytes per character, plus null terminator.
            let buf_size = len * 4 + 1;
            let mut buf: Vec<u8> = vec![0; buf_size as usize];
            if CFStringGetCString(cf, buf.as_mut_ptr() as *mut c_char, buf_size, kCFStringEncodingUTF8) {
                CStr::from_ptr(buf.as_ptr() as *const c_char)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            } else {
                None
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub fn read_focused_text_field() -> Option<String> {
    use std::ptr;

    unsafe {
        let system = ax::AXUIElementCreateSystemWide();
        if system.is_null() {
            return None;
        }

        // Get the focused UI element.
        let attr_focused = ax::cfstr("AXFocusedUIElement");
        let mut focused: ax::CFTypeRef = ptr::null();
        let err = ax::AXUIElementCopyAttributeValue(system, attr_focused, &mut focused);
        ax::CFRelease(attr_focused);

        if err != ax::kAXErrorSuccess || focused.is_null() {
            ax::CFRelease(system as *const _);
            return None;
        }

        // Get the text value of the focused element.
        let attr_value = ax::cfstr("AXValue");
        let mut value: ax::CFTypeRef = ptr::null();
        let err = ax::AXUIElementCopyAttributeValue(
            focused as ax::AXUIElementRef,
            attr_value,
            &mut value,
        );
        ax::CFRelease(attr_value);
        ax::CFRelease(focused);
        ax::CFRelease(system as *const _);

        if err != ax::kAXErrorSuccess || value.is_null() {
            return None;
        }

        let result = ax::cfstring_to_string(value as ax::CFStringRef);
        ax::CFRelease(value);

        // Truncate to the last MAX_CONTEXT_CHARS characters.
        result.map(|s| truncate_tail(&s, MAX_CONTEXT_CHARS))
    }
}

#[cfg(not(target_os = "macos"))]
pub fn read_focused_text_field() -> Option<String> {
    None
}

/// Keep only the last `max_chars` characters of a string.
fn truncate_tail(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        s.chars().skip(char_count - max_chars).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_tail_short() {
        assert_eq!(truncate_tail("hello", 500), "hello");
    }

    #[test]
    fn test_truncate_tail_exact() {
        let s = "a".repeat(500);
        assert_eq!(truncate_tail(&s, 500), s);
    }

    #[test]
    fn test_truncate_tail_long() {
        let s = "a".repeat(100) + &"b".repeat(500);
        let result = truncate_tail(&s, 500);
        assert_eq!(result.len(), 500);
        assert!(result.chars().all(|c| c == 'b'));
    }

    #[test]
    fn test_truncate_tail_empty() {
        assert_eq!(truncate_tail("", 500), "");
    }
}
