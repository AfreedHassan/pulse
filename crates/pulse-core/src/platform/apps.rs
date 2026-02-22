//! Application detection and context tracking.
//!
//! Provides macOS-specific frontmost app detection and maintains
//! a mapping of app names to categories for context-aware formatting.

use crate::types::{AppCategory, AppContext};

/// Detect the frontmost application on macOS.
///
/// Returns the app name and bundle ID if available.
#[cfg(target_os = "macos")]
pub fn detect_frontmost_app() -> Option<AppContext> {
    // Get app name.
    let name_output = std::process::Command::new("osascript")
        .args([
            "-e",
            "tell application \"System Events\" to get name of first application process whose frontmost is true",
        ])
        .output()
        .ok()?;

    let app_name = if name_output.status.success() {
        String::from_utf8_lossy(&name_output.stdout)
            .trim()
            .to_string()
    } else {
        return None;
    };

    if app_name.is_empty() {
        return None;
    }

    // Try to get bundle ID.
    let bundle_output = std::process::Command::new("osascript")
        .args([
            "-e",
            &format!("id of application \"{}\"", app_name.replace('"', "\\\"")),
        ])
        .output()
        .ok();

    let bundle_id = bundle_output.and_then(|o| {
        if o.status.success() {
            let id = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if id.is_empty() {
                None
            } else {
                Some(id)
            }
        } else {
            None
        }
    });

    // Try to get window title.
    let title_output = std::process::Command::new("osascript")
        .args([
            "-e",
            &format!(
                "tell application \"System Events\" to get title of front window of process \"{}\"",
                app_name.replace('"', "\\\"")
            ),
        ])
        .output()
        .ok();

    let window_title = title_output.and_then(|o| {
        if o.status.success() {
            let title = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if title.is_empty() {
                None
            } else {
                Some(title)
            }
        } else {
            None
        }
    });

    let category = AppCategory::from_app(&app_name, bundle_id.as_deref());

    Some(AppContext {
        app_name,
        bundle_id,
        window_title,
        category,
    })
}

/// Fallback for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub fn detect_frontmost_app() -> Option<AppContext> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_category_email() {
        let category = AppCategory::from_app("Mail", None);
        assert_eq!(category, AppCategory::Email);
    }

    #[test]
    fn test_app_category_code() {
        let category = AppCategory::from_app("Visual Studio Code", None);
        assert_eq!(category, AppCategory::Code);
    }

    #[test]
    fn test_app_category_unknown() {
        let category = AppCategory::from_app("SomeRandomApp", None);
        assert_eq!(category, AppCategory::Unknown);
    }
}
