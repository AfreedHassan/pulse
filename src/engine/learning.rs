//! Correction learning engine.
//!
//! Automatically learns from manual corrections and applies them to future
//! transcriptions. Uses fuzzy string matching (Levenshtein distance) to find
//! corrections that approximately match.

use strsim::levenshtein;

use crate::storage::Storage;
use crate::types::{Correction, CorrectionSource};

/// Post-processing engine that applies learned corrections to transcript text.
pub struct LearningEngine<'a> {
    storage: &'a Storage,
    corrections: Vec<Correction>,
    /// Max Levenshtein distance for fuzzy matching.
    max_distance: usize,
    /// Min confidence score to apply a correction.
    min_confidence: f32,
}

impl<'a> LearningEngine<'a> {
    pub fn new(storage: &'a Storage) -> Self {
        let corrections = storage.get_corrections(0.0).unwrap_or_default();
        Self {
            storage,
            corrections,
            max_distance: 1,
            min_confidence: 0.3,
        }
    }

    pub fn refresh(&mut self) {
        self.corrections = self
            .storage
            .get_corrections(self.min_confidence)
            .unwrap_or_default();
    }

    /// Record a new correction.
    pub fn learn(&mut self, original: &str, corrected: &str) {
        let c = Correction::new(
            original.to_string(),
            corrected.to_string(),
            CorrectionSource::UserEdit,
        );
        if self.storage.save_correction(&c).is_ok() {
            self.refresh();
        }
    }

    /// Apply all matched corrections to the text.
    pub fn apply(&self, text: &str) -> String {
        let mut result = text.to_string();

        for correction in &self.corrections {
            if correction.confidence < self.min_confidence {
                continue;
            }

            // Direct replacement (case-insensitive).
            if let Some(idx) = result
                .to_lowercase()
                .find(&correction.original.to_lowercase())
            {
                let end = idx + correction.original.len();
                result = format!(
                    "{}{}{}",
                    &result[..idx],
                    &correction.corrected,
                    &result[end..]
                );
                continue;
            }

            // Fuzzy match individual words.
            let target_words: Vec<&str> = correction.original.split_whitespace().collect();
            if target_words.len() == 1 {
                let words: Vec<String> = result
                    .split_whitespace()
                    .map(|w| {
                        if levenshtein(&w.to_lowercase(), &target_words[0].to_lowercase())
                            <= self.max_distance
                            && !w.is_empty()
                        {
                            correction.corrected.clone()
                        } else {
                            w.to_string()
                        }
                    })
                    .collect();
                result = words.join(" ");
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_correction(storage: &Storage, original: &str, corrected: &str) {
        let c = Correction::new(
            original.to_string(),
            corrected.to_string(),
            CorrectionSource::UserEdit,
        );
        storage.save_correction(&c).unwrap();
    }

    fn make_correction_with_confidence(
        storage: &Storage,
        original: &str,
        corrected: &str,
        confidence: f32,
    ) {
        let mut c = Correction::new(
            original.to_string(),
            corrected.to_string(),
            CorrectionSource::UserEdit,
        );
        c.confidence = confidence;
        storage.save_correction(&c).unwrap();
    }

    #[test]
    fn test_direct_replacement() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "teh", "the");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0; // Accept any confidence for testing.
        engine.refresh();
        assert_eq!(engine.apply("teh quick brown fox"), "the quick brown fox");
    }

    #[test]
    fn test_fuzzy_replacement() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "recieve", "receive");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.refresh();
        // "recieve" is distance 0 (exact match substring).
        assert_eq!(engine.apply("I recieve the email"), "I receive the email");
    }

    #[test]
    fn test_no_match() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "teh", "the");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.refresh();
        // "text" has distance 2 from "teh", which is > max_distance(1), so no match.
        assert_eq!(
            engine.apply("completely different text"),
            "completely different text"
        );
    }

    #[test]
    fn test_learn_and_apply() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.learn("hte", "the");
        assert_eq!(engine.apply("hte cat"), "the cat");
    }

    #[test]
    fn test_case_insensitive_replacement() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "teh", "the");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.refresh();
        assert_eq!(engine.apply("TEH QUICK BROWN FOX"), "the QUICK BROWN FOX");
    }

    #[test]
    fn test_multiple_corrections() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "teh", "the");
        make_correction(&storage, "adn", "and");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.refresh();
        assert_eq!(engine.apply("teh cat adn dog"), "the cat and dog");
    }

    #[test]
    fn test_confidence_filter() {
        let storage = Storage::in_memory().unwrap();
        make_correction_with_confidence(&storage, "teh", "the", 0.2);
        make_correction_with_confidence(&storage, "adn", "and", 0.8);

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.5;
        engine.refresh();
        // Only "adn" should be applied
        assert_eq!(engine.apply("teh cat adn dog"), "teh cat and dog");
    }

    #[test]
    fn test_empty_text() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "teh", "the");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.refresh();
        assert_eq!(engine.apply(""), "");
    }

    #[test]
    fn test_no_corrections() {
        let storage = Storage::in_memory().unwrap();
        let engine = LearningEngine::new(&storage);
        assert_eq!(engine.apply("hello world"), "hello world");
    }

    #[test]
    fn test_fuzzy_match_typo() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "receive", "receive");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.max_distance = 2;
        engine.refresh();
        // "recieve" is distance 2 from "receive" (swap of e and i)
        assert_eq!(engine.apply("I recieve it"), "I receive it");
    }

    #[test]
    fn test_fuzzy_respects_max_distance() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "hello", "hi");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.max_distance = 1;
        engine.refresh();
        // "hallo" is distance 1 from "hello" (should match)
        assert_eq!(engine.apply("hallo world"), "hi world");
        // "hullo" is distance 1 from "hello" (should match)
        assert_eq!(engine.apply("hullo world"), "hi world");
    }

    #[test]
    fn test_multi_word_correction_direct() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "see you later", "goodbye");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.refresh();
        // Multi-word corrections do direct replacement
        assert_eq!(
            engine.apply("I will see you later now"),
            "I will goodbye now"
        );
    }

    #[test]
    fn test_correction_at_start() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "teh", "the");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.refresh();
        assert_eq!(engine.apply("teh end"), "the end");
    }

    #[test]
    fn test_correction_at_end() {
        let storage = Storage::in_memory().unwrap();
        make_correction(&storage, "teh", "the");

        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;
        engine.refresh();
        assert_eq!(engine.apply("at teh"), "at the");
    }

    #[test]
    fn test_refresh_reloads_corrections() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = LearningEngine::new(&storage);
        engine.min_confidence = 0.0;

        // Initially no corrections
        assert_eq!(engine.apply("teh cat"), "teh cat");

        // Add a correction directly to storage
        make_correction(&storage, "teh", "the");

        // Refresh should reload
        engine.refresh();
        assert_eq!(engine.apply("teh cat"), "the cat");
    }
}
