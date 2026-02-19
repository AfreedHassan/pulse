//! Shortcut expansion engine.
//!
//! Expands user-defined text shortcuts in transcriptions using
//! Aho-Corasick for O(n) multi-pattern matching.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

use crate::storage::Storage;
use crate::types::Shortcut;

/// Shortcut expansion engine using Aho-Corasick for parallel matching.
pub struct ShortcutEngine {
    shortcuts: Vec<Shortcut>,
    automaton: Option<AhoCorasick>,
}

impl ShortcutEngine {
    pub fn new(storage: &Storage) -> Self {
        let shortcuts = storage.get_enabled_shortcuts().unwrap_or_default();
        let automaton = Self::build_automaton(&shortcuts);
        Self {
            shortcuts,
            automaton,
        }
    }

    fn build_automaton(shortcuts: &[Shortcut]) -> Option<AhoCorasick> {
        if shortcuts.is_empty() {
            return None;
        }

        let patterns: Vec<&str> = shortcuts.iter().map(|s| s.trigger.as_str()).collect();
        AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns)
            .ok()
    }

    pub fn refresh(&mut self, storage: &Storage) {
        self.shortcuts = storage.get_enabled_shortcuts().unwrap_or_default();
        self.automaton = Self::build_automaton(&self.shortcuts);
    }

    /// Expand all shortcuts in the text.
    pub fn expand(&self, text: &str) -> String {
        let automaton = match &self.automaton {
            Some(a) => a,
            None => return text.to_string(),
        };

        let mut replacements: Vec<(usize, usize, &str)> = Vec::new();
        for mat in automaton.find_iter(text) {
            let shortcut = &self.shortcuts[mat.pattern()];
            replacements.push((mat.start(), mat.end(), &shortcut.replacement));
        }

        if replacements.is_empty() {
            return text.to_string();
        }

        let mut result = String::with_capacity(text.len() * 2);
        let mut last_end = 0;

        for (start, end, replacement) in &replacements {
            result.push_str(&text[last_end..*start]);
            result.push_str(replacement);
            last_end = *end;
        }
        result.push_str(&text[last_end..]);

        result
    }

    pub fn len(&self) -> usize {
        self.shortcuts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shortcuts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_shortcut(storage: &Storage, trigger: &str, replacement: &str) {
        let s = Shortcut::new(trigger.to_string(), replacement.to_string());
        storage.save_shortcut(&s).unwrap();
    }

    fn add_disabled_shortcut(storage: &Storage, trigger: &str, replacement: &str) {
        let mut s = Shortcut::new(trigger.to_string(), replacement.to_string());
        s.enabled = false;
        storage.save_shortcut(&s).unwrap();
    }

    #[test]
    fn test_single_expansion() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("I'll brb"), "I'll be right back");
    }

    #[test]
    fn test_multiple_expansions() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");
        add_shortcut(&storage, "tbh", "to be honest");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(
            engine.expand("tbh I'll brb"),
            "to be honest I'll be right back"
        );
    }

    #[test]
    fn test_no_shortcuts() {
        let storage = Storage::in_memory().unwrap();
        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("hello world"), "hello world");
        assert!(engine.is_empty());
    }

    #[test]
    fn test_disabled_shortcut() {
        let storage = Storage::in_memory().unwrap();
        add_disabled_shortcut(&storage, "brb", "be right back");

        let engine = ShortcutEngine::new(&storage);
        // Disabled shortcuts should NOT be expanded.
        assert_eq!(engine.expand("I'll brb"), "I'll brb");
    }

    #[test]
    fn test_no_match_in_text() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("hello world"), "hello world");
    }

    #[test]
    fn test_expansion_at_start() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("brb soon"), "be right back soon");
    }

    #[test]
    fn test_expansion_at_end() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("I'll brb"), "I'll be right back");
    }

    #[test]
    fn test_expansion_entire_text() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("brb"), "be right back");
    }

    #[test]
    fn test_expansion_with_longer_replacement() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "addr", "123 Main Street, Anytown, USA 12345");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(
            engine.expand("My addr is here"),
            "My 123 Main Street, Anytown, USA 12345 is here"
        );
    }

    #[test]
    fn test_expansion_with_shorter_replacement() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "asap", "now");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("I need it asap"), "I need it now");
    }

    #[test]
    fn test_same_trigger_multiple_times() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(
            engine.expand("brb brb brb"),
            "be right back be right back be right back"
        );
    }

    #[test]
    fn test_empty_text() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand(""), "");
    }

    #[test]
    fn test_refresh_reloads_shortcuts() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ShortcutEngine::new(&storage);

        // Initially no shortcuts
        assert_eq!(engine.expand("brb"), "brb");

        // Add a shortcut
        add_shortcut(&storage, "brb", "be right back");

        // Refresh should reload
        engine.refresh(&storage);
        assert_eq!(engine.expand("brb"), "be right back");
    }

    #[test]
    fn test_len_and_is_empty() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ShortcutEngine::new(&storage);

        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);

        add_shortcut(&storage, "brb", "be right back");
        engine.refresh(&storage);

        assert!(!engine.is_empty());
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn test_overlapping_patterns_leftmost_longest() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "my email", "user@example.com");
        add_shortcut(&storage, "my", "the");

        let engine = ShortcutEngine::new(&storage);
        // "my email" should match as a whole (leftmost-longest)
        let result = engine.expand("send to my email");
        assert_eq!(result, "send to user@example.com");
    }

    #[test]
    fn test_trigger_with_spaces() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "my address", "123 Main St");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("my address is here"), "123 Main St is here");
    }

    #[test]
    fn test_mixed_enabled_disabled() {
        let storage = Storage::in_memory().unwrap();
        add_shortcut(&storage, "brb", "be right back");
        add_disabled_shortcut(&storage, "tbh", "to be honest");

        let engine = ShortcutEngine::new(&storage);
        assert_eq!(engine.expand("brb tbh"), "be right back tbh");
    }
}
