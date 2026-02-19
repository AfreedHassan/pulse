//! App context types for context-aware transcription.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppCategory {
    Email,
    Slack,
    Imessage,
    Code,
    Documents,
    Social,
    Browser,
    Terminal,
    #[default]
    Unknown,
}

impl AppCategory {
    pub fn from_app(app_name: &str, bundle_id: Option<&str>) -> Self {
        let name_lower = app_name.to_lowercase();
        let bundle_lower = bundle_id.map(|b| b.to_lowercase()).unwrap_or_default();

        if name_lower.contains("mail") || bundle_lower.contains("mail") {
            AppCategory::Email
        } else if name_lower.contains("slack")
            || name_lower.contains("discord")
            || name_lower.contains("teams")
        {
            AppCategory::Slack
        } else if name_lower.contains("imessage")
            || name_lower.contains("messages")
            || bundle_lower.contains("imessage")
            || bundle_lower.contains("messages")
        {
            AppCategory::Imessage
        } else if name_lower.contains("code")
            || name_lower.contains("xcode")
            || name_lower.contains("intellij")
            || name_lower.contains("vim")
            || name_lower.contains("nvim")
            || name_lower.contains("cursor")
            || name_lower.contains("zed")
        {
            AppCategory::Code
        } else if name_lower.contains("pages")
            || name_lower.contains("word")
            || name_lower.contains("docs")
            || name_lower.contains("notion")
            || name_lower.contains("obsidian")
        {
            AppCategory::Documents
        } else if name_lower.contains("twitter")
            || name_lower.contains("facebook")
            || name_lower.contains("instagram")
        {
            AppCategory::Social
        } else if name_lower.contains("safari")
            || name_lower.contains("chrome")
            || name_lower.contains("firefox")
            || name_lower.contains("arc")
        {
            AppCategory::Browser
        } else if name_lower.contains("terminal")
            || name_lower.contains("iterm")
            || name_lower.contains("warp")
            || name_lower.contains("kitty")
            || name_lower.contains("alacritty")
            || name_lower.contains("ghostty")
        {
            AppCategory::Terminal
        } else {
            AppCategory::Unknown
        }
    }

    pub fn all() -> &'static [AppCategory] {
        &[
            AppCategory::Email,
            AppCategory::Slack,
            AppCategory::Imessage,
            AppCategory::Code,
            AppCategory::Documents,
            AppCategory::Social,
            AppCategory::Browser,
            AppCategory::Terminal,
            AppCategory::Unknown,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppContext {
    pub app_name: String,
    pub bundle_id: Option<String>,
    pub window_title: Option<String>,
    pub category: AppCategory,
}

impl Default for AppContext {
    fn default() -> Self {
        Self {
            app_name: "Unknown".to_string(),
            bundle_id: None,
            window_title: None,
            category: AppCategory::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_category_from_app_email() {
        assert_eq!(AppCategory::from_app("Mail", None), AppCategory::Email);
        assert_eq!(AppCategory::from_app("Gmail", None), AppCategory::Email);
    }

    #[test]
    fn test_app_category_from_app_code() {
        assert_eq!(
            AppCategory::from_app("Visual Studio Code", None),
            AppCategory::Code
        );
        assert_eq!(AppCategory::from_app("Cursor", None), AppCategory::Code);
        assert_eq!(AppCategory::from_app("Xcode", None), AppCategory::Code);
        assert_eq!(AppCategory::from_app("Zed", None), AppCategory::Code);
    }

    #[test]
    fn test_app_category_from_app_slack() {
        assert_eq!(AppCategory::from_app("Slack", None), AppCategory::Slack);
        assert_eq!(AppCategory::from_app("Discord", None), AppCategory::Slack);
        assert_eq!(
            AppCategory::from_app("Microsoft Teams", None),
            AppCategory::Slack
        );
    }

    #[test]
    fn test_app_category_from_app_imessage() {
        assert_eq!(
            AppCategory::from_app("Messages", None),
            AppCategory::Imessage
        );
        assert_eq!(
            AppCategory::from_app("iMessage", None),
            AppCategory::Imessage
        );
        assert_eq!(
            AppCategory::from_app("SomeApp", Some("com.apple.messages")),
            AppCategory::Imessage
        );
    }

    #[test]
    fn test_app_category_from_app_browser() {
        assert_eq!(AppCategory::from_app("Safari", None), AppCategory::Browser);
        assert_eq!(
            AppCategory::from_app("Google Chrome", None),
            AppCategory::Browser
        );
        assert_eq!(AppCategory::from_app("Firefox", None), AppCategory::Browser);
        assert_eq!(AppCategory::from_app("Arc", None), AppCategory::Browser);
    }

    #[test]
    fn test_app_category_from_app_terminal() {
        assert_eq!(
            AppCategory::from_app("Terminal", None),
            AppCategory::Terminal
        );
        assert_eq!(AppCategory::from_app("iTerm2", None), AppCategory::Terminal);
        assert_eq!(AppCategory::from_app("Warp", None), AppCategory::Terminal);
        assert_eq!(AppCategory::from_app("Kitty", None), AppCategory::Terminal);
        assert_eq!(
            AppCategory::from_app("Ghostty", None),
            AppCategory::Terminal
        );
        assert_eq!(
            AppCategory::from_app("Alacritty", None),
            AppCategory::Terminal
        );
    }

    #[test]
    fn test_app_category_from_app_documents() {
        assert_eq!(AppCategory::from_app("Pages", None), AppCategory::Documents);
        assert_eq!(
            AppCategory::from_app("Notion", None),
            AppCategory::Documents
        );
        assert_eq!(
            AppCategory::from_app("Obsidian", None),
            AppCategory::Documents
        );
    }

    #[test]
    fn test_app_category_from_app_social() {
        assert_eq!(AppCategory::from_app("Twitter", None), AppCategory::Social);
        assert_eq!(
            AppCategory::from_app("Instagram", None),
            AppCategory::Social
        );
    }

    #[test]
    fn test_app_category_from_app_unknown() {
        assert_eq!(
            AppCategory::from_app("SomeRandomApp", None),
            AppCategory::Unknown
        );
    }

    #[test]
    fn test_app_category_from_bundle_id() {
        assert_eq!(
            AppCategory::from_app("SomeApp", Some("com.apple.mail")),
            AppCategory::Email
        );
    }

    #[test]
    fn test_app_category_all() {
        let all = AppCategory::all();
        assert_eq!(all.len(), 9);
        assert!(all.contains(&AppCategory::Email));
        assert!(all.contains(&AppCategory::Unknown));
    }

    #[test]
    fn test_app_context_default() {
        let ctx = AppContext::default();
        assert_eq!(ctx.app_name, "Unknown");
        assert_eq!(ctx.category, AppCategory::Unknown);
        assert!(ctx.bundle_id.is_none());
        assert!(ctx.window_title.is_none());
    }

    #[test]
    fn test_app_category_serde() {
        let cat = AppCategory::Email;
        let json = serde_json::to_string(&cat).unwrap();
        assert_eq!(json, "\"email\"");
        let decoded: AppCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, cat);
    }

    #[test]
    fn test_app_context_serde() {
        let ctx = AppContext {
            app_name: "TestApp".to_string(),
            bundle_id: Some("com.test.app".to_string()),
            window_title: Some("Test Window".to_string()),
            category: AppCategory::Code,
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let decoded: AppContext = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.app_name, "TestApp");
        assert_eq!(decoded.category, AppCategory::Code);
    }
}
