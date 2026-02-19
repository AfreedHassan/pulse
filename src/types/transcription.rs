//! Transcription types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::app::AppContext;

pub type TranscriptionId = Uuid;

pub type AudioData = Vec<u8>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcription {
    pub id: TranscriptionId,
    pub raw_text: String,
    pub processed_text: String,
    pub confidence: f32,
    pub duration_ms: u64,
    pub app_context: Option<AppContext>,
    pub created_at: DateTime<Utc>,
}

impl Transcription {
    pub fn new(
        raw_text: String,
        processed_text: String,
        confidence: f32,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            raw_text,
            processed_text,
            confidence,
            duration_ms,
            app_context: None,
            created_at: Utc::now(),
        }
    }

    pub fn with_context(mut self, context: AppContext) -> Self {
        self.app_context = Some(context);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionStatus {
    Success,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionHistoryEntry {
    pub id: TranscriptionId,
    pub status: TranscriptionStatus,
    pub text: String,
    pub raw_text: String,
    pub error: Option<String>,
    pub duration_ms: u64,
    pub app_context: Option<AppContext>,
    pub created_at: DateTime<Utc>,
}

impl TranscriptionHistoryEntry {
    pub fn success(raw_text: String, text: String, duration_ms: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            status: TranscriptionStatus::Success,
            text,
            raw_text,
            error: None,
            duration_ms,
            app_context: None,
            created_at: Utc::now(),
        }
    }

    pub fn failure(error: String, duration_ms: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            status: TranscriptionStatus::Failed,
            text: String::new(),
            raw_text: String::new(),
            error: Some(error),
            duration_ms,
            app_context: None,
            created_at: Utc::now(),
        }
    }

    pub fn with_context(mut self, context: AppContext) -> Self {
        self.app_context = Some(context);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::app::{AppCategory, AppContext};

    #[test]
    fn test_transcription_new() {
        let t = Transcription::new("hello".into(), "Hello.".into(), 0.95, 1200);
        assert_eq!(t.raw_text, "hello");
        assert_eq!(t.processed_text, "Hello.");
        assert_eq!(t.confidence, 0.95);
        assert_eq!(t.duration_ms, 1200);
        assert!(t.app_context.is_none());
        assert!(!t.id.is_nil());
    }

    #[test]
    fn test_transcription_with_context() {
        let ctx = AppContext {
            app_name: "TestApp".to_string(),
            bundle_id: None,
            window_title: None,
            category: AppCategory::Code,
        };
        let t = Transcription::new("hello".into(), "Hello.".into(), 0.95, 1200)
            .with_context(ctx.clone());
        assert!(t.app_context.is_some());
        assert_eq!(t.app_context.unwrap().app_name, "TestApp");
    }

    #[test]
    fn test_transcription_history_success() {
        let e = TranscriptionHistoryEntry::success("raw".into(), "text".into(), 500);
        assert_eq!(e.status, TranscriptionStatus::Success);
        assert!(e.error.is_none());
        assert_eq!(e.raw_text, "raw");
        assert_eq!(e.text, "text");
        assert_eq!(e.duration_ms, 500);
        assert!(!e.id.is_nil());
    }

    #[test]
    fn test_transcription_history_failure() {
        let e = TranscriptionHistoryEntry::failure("boom".into(), 100);
        assert_eq!(e.status, TranscriptionStatus::Failed);
        assert_eq!(e.error, Some("boom".into()));
        assert!(e.text.is_empty());
        assert!(e.raw_text.is_empty());
    }

    #[test]
    fn test_transcription_history_with_context() {
        let ctx = AppContext {
            app_name: "Slack".to_string(),
            bundle_id: None,
            window_title: None,
            category: AppCategory::Slack,
        };
        let e =
            TranscriptionHistoryEntry::success("hi".into(), "Hi!".into(), 100).with_context(ctx);
        assert!(e.app_context.is_some());
    }

    #[test]
    fn test_transcription_status_serde() {
        let status = TranscriptionStatus::Success;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"success\"");
        let decoded: TranscriptionStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, status);

        let failed = TranscriptionStatus::Failed;
        let json = serde_json::to_string(&failed).unwrap();
        assert_eq!(json, "\"failed\"");
    }

    #[test]
    fn test_audio_data_type() {
        let audio: AudioData = vec![0u8, 1, 2, 3];
        assert_eq!(audio.len(), 4);
    }

    #[test]
    fn test_transcription_serde() {
        let t = Transcription::new("hello world".into(), "Hello, world.".into(), 0.9, 1500);
        let json = serde_json::to_string(&t).unwrap();
        let decoded: Transcription = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.raw_text, "hello world");
        assert_eq!(decoded.processed_text, "Hello, world.");
    }
}
