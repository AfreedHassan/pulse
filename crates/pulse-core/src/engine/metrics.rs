//! Analytics and metrics collection.
//!
//! Tracks usage patterns to help improve transcription quality and
//! user experience over time. All data stays local.

use crate::storage::Storage;
use crate::types::{AnalyticsEvent, EventType};

/// Metrics collector for transcription analytics.
pub struct Metrics<'a> {
    storage: &'a Storage,
}

impl<'a> Metrics<'a> {
    pub fn new(storage: &'a Storage) -> Self {
        Self { storage }
    }

    /// Record a transcription completed event.
    pub fn record_transcription(&self, duration_ms: u64, word_count: usize, provider: &str) {
        let event = AnalyticsEvent::new(
            EventType::TranscriptionCompleted,
            serde_json::json!({
                "duration_ms": duration_ms,
                "word_count": word_count,
                "provider": provider,
            }),
        );
        let _ = self.storage.save_event(&event);
    }

    /// Record a transcription failure.
    pub fn record_transcription_failed(&self, error: &str) {
        let event = AnalyticsEvent::new(
            EventType::TranscriptionFailed,
            serde_json::json!({
                "error": error,
            }),
        );
        let _ = self.storage.save_event(&event);
    }

    /// Record a shortcut expansion event.
    pub fn record_shortcut_usage(&self, trigger: &str) {
        let event = AnalyticsEvent::new(
            EventType::ShortcutTriggered,
            serde_json::json!({
                "trigger": trigger,
            }),
        );
        let _ = self.storage.save_event(&event);
    }

    /// Record a correction application event.
    pub fn record_correction(&self, original: &str, corrected: &str) {
        let event = AnalyticsEvent::new(
            EventType::CorrectionApplied,
            serde_json::json!({
                "original": original,
                "corrected": corrected,
            }),
        );
        let _ = self.storage.save_event(&event);
    }

    /// Record a mode change event.
    pub fn record_mode_change(&self, app: &str, mode: &str) {
        let event = AnalyticsEvent::new(
            EventType::ModeChanged,
            serde_json::json!({
                "app": app,
                "mode": mode,
            }),
        );
        let _ = self.storage.save_event(&event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_transcription() {
        let storage = Storage::in_memory().unwrap();
        let metrics = Metrics::new(&storage);
        metrics.record_transcription(1500, 25, "local-pulse");
    }

    #[test]
    fn test_record_failure() {
        let storage = Storage::in_memory().unwrap();
        let metrics = Metrics::new(&storage);
        metrics.record_transcription_failed("audio too short");
    }

    #[test]
    fn test_record_shortcut() {
        let storage = Storage::in_memory().unwrap();
        let metrics = Metrics::new(&storage);
        metrics.record_shortcut_usage("brb");
    }

    #[test]
    fn test_record_correction() {
        let storage = Storage::in_memory().unwrap();
        let metrics = Metrics::new(&storage);
        metrics.record_correction("teh", "the");
    }

    #[test]
    fn test_record_mode_change() {
        let storage = Storage::in_memory().unwrap();
        let metrics = Metrics::new(&storage);
        metrics.record_mode_change("Mail", "formal");
    }
}
