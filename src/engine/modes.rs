//! App-aware writing mode engine.
//!
//! Automatically detects the frontmost application and selects an appropriate
//! writing mode based on user preferences stored in the database.

use crate::platform::apps::detect_frontmost_app;
use crate::storage::Storage;
use crate::types::{AppContext, WritingMode};

/// Mode engine that selects writing mode based on app context.
pub struct ModeEngine<'a> {
    storage: &'a Storage,
}

impl<'a> ModeEngine<'a> {
    pub fn new(storage: &'a Storage) -> Self {
        Self { storage }
    }

    /// Determine the best writing mode for the given app context.
    ///
    /// Priority:
    /// 1. User-configured mode for this specific app (from storage).
    /// 2. Category-based default.
    /// 3. Global default (Casual).
    pub fn resolve_mode(&self, context: &AppContext) -> WritingMode {
        if let Ok(Some(mode)) = self.storage.get_app_mode(&context.app_name) {
            return mode;
        }
        WritingMode::suggested_for_category(context.category)
    }

    /// Set a user-preferred mode for a specific app.
    pub fn set_app_mode(&self, app_name: &str, mode: WritingMode) -> crate::error::Result<()> {
        self.storage.save_app_mode(app_name, mode)
    }

    /// Get the current context for the frontmost app (macOS only).
    pub fn current_context(&self) -> AppContext {
        #[cfg(target_os = "macos")]
        {
            detect_frontmost_app().unwrap_or_default()
        }

        #[cfg(not(target_os = "macos"))]
        {
            AppContext::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AppCategory;

    fn ctx(app_name: &str, category: AppCategory) -> AppContext {
        AppContext {
            app_name: app_name.to_string(),
            bundle_id: None,
            window_title: None,
            category,
        }
    }

    #[test]
    fn test_resolve_mode_default() {
        let storage = Storage::in_memory().unwrap();
        let engine = ModeEngine::new(&storage);
        assert_eq!(
            engine.resolve_mode(&ctx("Unknown", AppCategory::Unknown)),
            WritingMode::Casual
        );
    }

    #[test]
    fn test_resolve_mode_category_default() {
        let storage = Storage::in_memory().unwrap();
        let engine = ModeEngine::new(&storage);
        assert_eq!(
            engine.resolve_mode(&ctx("Mail", AppCategory::Email)),
            WritingMode::Formal
        );
    }

    #[test]
    fn test_resolve_mode_user_override() {
        let storage = Storage::in_memory().unwrap();
        let engine = ModeEngine::new(&storage);

        engine.set_app_mode("Mail", WritingMode::Casual).unwrap();

        assert_eq!(
            engine.resolve_mode(&ctx("Mail", AppCategory::Email)),
            WritingMode::Casual
        );
    }

    #[test]
    fn test_resolve_mode_slack() {
        let storage = Storage::in_memory().unwrap();
        let engine = ModeEngine::new(&storage);
        assert_eq!(
            engine.resolve_mode(&ctx("Slack", AppCategory::Slack)),
            WritingMode::Casual
        );
    }

    #[test]
    fn test_resolve_mode_terminal() {
        let storage = Storage::in_memory().unwrap();
        let engine = ModeEngine::new(&storage);
        assert_eq!(
            engine.resolve_mode(&ctx("Terminal", AppCategory::Terminal)),
            WritingMode::VeryCasual
        );
    }

    #[test]
    fn test_resolve_mode_code() {
        let storage = Storage::in_memory().unwrap();
        let engine = ModeEngine::new(&storage);
        assert_eq!(
            engine.resolve_mode(&ctx("VS Code", AppCategory::Code)),
            WritingMode::Formal
        );
    }

    #[test]
    fn test_current_context() {
        let storage = Storage::in_memory().unwrap();
        let engine = ModeEngine::new(&storage);
        let ctx = engine.current_context();
        assert!(!ctx.app_name.is_empty());
    }
}
