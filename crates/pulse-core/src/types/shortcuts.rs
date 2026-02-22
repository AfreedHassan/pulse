//! Shortcut types for text expansion.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type ShortcutId = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shortcut {
    pub id: ShortcutId,
    pub trigger: String,
    pub replacement: String,
    pub case_sensitive: bool,
    pub enabled: bool,
    pub use_count: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Shortcut {
    pub fn new(trigger: String, replacement: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            trigger,
            replacement,
            case_sensitive: false,
            enabled: true,
            use_count: 0,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_case_sensitive(mut self, sensitive: bool) -> Self {
        self.case_sensitive = sensitive;
        self
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn increment_use(&mut self) {
        self.use_count += 1;
        self.updated_at = Utc::now();
    }

    pub fn matches(&self, text: &str) -> bool {
        if !self.enabled {
            return false;
        }
        if self.case_sensitive {
            text.contains(&self.trigger)
        } else {
            text.to_lowercase().contains(&self.trigger.to_lowercase())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortcut_new() {
        let s = Shortcut::new("my email".into(), "user@example.com".into());
        assert_eq!(s.trigger, "my email");
        assert_eq!(s.replacement, "user@example.com");
        assert!(!s.case_sensitive);
        assert!(s.enabled);
        assert_eq!(s.use_count, 0);
        assert!(!s.id.is_nil());
    }

    #[test]
    fn test_shortcut_with_case_sensitive() {
        let s = Shortcut::new("BRB".into(), "Be Right Back".into()).with_case_sensitive(true);
        assert!(s.case_sensitive);
    }

    #[test]
    fn test_shortcut_with_enabled() {
        let s = Shortcut::new("test".into(), "replacement".into()).with_enabled(false);
        assert!(!s.enabled);
    }

    #[test]
    fn test_shortcut_increment_use() {
        let mut s = Shortcut::new("test".into(), "replacement".into());
        assert_eq!(s.use_count, 0);
        s.increment_use();
        assert_eq!(s.use_count, 1);
        s.increment_use();
        assert_eq!(s.use_count, 2);
    }

    #[test]
    fn test_shortcut_matches_case_insensitive() {
        let s = Shortcut::new("brb".into(), "be right back".into());
        assert!(s.matches("I'll brb"));
        assert!(s.matches("I'll BRB"));
        assert!(s.matches("BRB everyone"));
        assert!(!s.matches("see you"));
    }

    #[test]
    fn test_shortcut_matches_case_sensitive() {
        let s = Shortcut::new("BRB".into(), "Be Right Back".into()).with_case_sensitive(true);
        assert!(s.matches("I'll BRB"));
        assert!(!s.matches("I'll brb"));
    }

    #[test]
    fn test_shortcut_matches_disabled() {
        let s = Shortcut::new("brb".into(), "be right back".into()).with_enabled(false);
        assert!(!s.matches("I'll brb"));
    }

    #[test]
    fn test_shortcut_serde() {
        let s = Shortcut::new("trigger".into(), "replacement".into());
        let json = serde_json::to_string(&s).unwrap();
        let decoded: Shortcut = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.trigger, "trigger");
        assert_eq!(decoded.replacement, "replacement");
    }
}
