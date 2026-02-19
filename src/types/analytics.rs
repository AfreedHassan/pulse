//! Analytics event types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::app::AppContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    TranscriptionStarted,
    TranscriptionCompleted,
    TranscriptionFailed,
    ShortcutTriggered,
    CorrectionApplied,
    ModeChanged,
    AppSwitched,
}

impl EventType {
    pub fn all() -> &'static [EventType] {
        &[
            EventType::TranscriptionStarted,
            EventType::TranscriptionCompleted,
            EventType::TranscriptionFailed,
            EventType::ShortcutTriggered,
            EventType::CorrectionApplied,
            EventType::ModeChanged,
            EventType::AppSwitched,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEvent {
    pub id: Uuid,
    pub event_type: EventType,
    pub properties: serde_json::Value,
    pub app_context: Option<AppContext>,
    pub created_at: DateTime<Utc>,
}

impl AnalyticsEvent {
    pub fn new(event_type: EventType, properties: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            properties,
            app_context: None,
            created_at: Utc::now(),
        }
    }

    pub fn with_context(mut self, context: AppContext) -> Self {
        self.app_context = Some(context);
        self
    }

    pub fn transcription_started() -> Self {
        Self::new(EventType::TranscriptionStarted, serde_json::json!({}))
    }

    pub fn transcription_completed(duration_ms: u64, word_count: usize, provider: &str) -> Self {
        Self::new(
            EventType::TranscriptionCompleted,
            serde_json::json!({
                "duration_ms": duration_ms,
                "word_count": word_count,
                "provider": provider,
            }),
        )
    }

    pub fn transcription_failed(error: &str) -> Self {
        Self::new(
            EventType::TranscriptionFailed,
            serde_json::json!({
                "error": error,
            }),
        )
    }

    pub fn shortcut_triggered(trigger: &str) -> Self {
        Self::new(
            EventType::ShortcutTriggered,
            serde_json::json!({
                "trigger": trigger,
            }),
        )
    }

    pub fn correction_applied(original: &str, corrected: &str) -> Self {
        Self::new(
            EventType::CorrectionApplied,
            serde_json::json!({
                "original": original,
                "corrected": corrected,
            }),
        )
    }

    pub fn mode_changed(app: &str, mode: &str) -> Self {
        Self::new(
            EventType::ModeChanged,
            serde_json::json!({
                "app": app,
                "mode": mode,
            }),
        )
    }

    pub fn app_switched(from_app: &str, to_app: &str) -> Self {
        Self::new(
            EventType::AppSwitched,
            serde_json::json!({
                "from_app": from_app,
                "to_app": to_app,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::app::{AppCategory, AppContext};

    #[test]
    fn test_event_type_all() {
        let all = EventType::all();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_analytics_event_new() {
        let e = AnalyticsEvent::new(
            EventType::TranscriptionCompleted,
            serde_json::json!({"duration_ms": 1500}),
        );
        assert_eq!(e.event_type, EventType::TranscriptionCompleted);
        assert!(e.app_context.is_none());
        assert!(!e.id.is_nil());
    }

    #[test]
    fn test_analytics_event_with_context() {
        let ctx = AppContext {
            app_name: "TestApp".to_string(),
            bundle_id: None,
            window_title: None,
            category: AppCategory::Code,
        };
        let e = AnalyticsEvent::transcription_completed(1000, 10, "local").with_context(ctx);
        assert!(e.app_context.is_some());
    }

    #[test]
    fn test_analytics_event_transcription_started() {
        let e = AnalyticsEvent::transcription_started();
        assert_eq!(e.event_type, EventType::TranscriptionStarted);
    }

    #[test]
    fn test_analytics_event_transcription_completed() {
        let e = AnalyticsEvent::transcription_completed(1500, 25, "local-pulse");
        assert_eq!(e.event_type, EventType::TranscriptionCompleted);
        let props = e.properties.as_object().unwrap();
        assert_eq!(props.get("duration_ms").unwrap().as_u64().unwrap(), 1500);
        assert_eq!(props.get("word_count").unwrap().as_u64().unwrap(), 25);
        assert_eq!(
            props.get("provider").unwrap().as_str().unwrap(),
            "local-pulse"
        );
    }

    #[test]
    fn test_analytics_event_transcription_failed() {
        let e = AnalyticsEvent::transcription_failed("audio too short");
        assert_eq!(e.event_type, EventType::TranscriptionFailed);
        let props = e.properties.as_object().unwrap();
        assert_eq!(
            props.get("error").unwrap().as_str().unwrap(),
            "audio too short"
        );
    }

    #[test]
    fn test_analytics_event_shortcut_triggered() {
        let e = AnalyticsEvent::shortcut_triggered("brb");
        assert_eq!(e.event_type, EventType::ShortcutTriggered);
        let props = e.properties.as_object().unwrap();
        assert_eq!(props.get("trigger").unwrap().as_str().unwrap(), "brb");
    }

    #[test]
    fn test_analytics_event_correction_applied() {
        let e = AnalyticsEvent::correction_applied("teh", "the");
        assert_eq!(e.event_type, EventType::CorrectionApplied);
        let props = e.properties.as_object().unwrap();
        assert_eq!(props.get("original").unwrap().as_str().unwrap(), "teh");
        assert_eq!(props.get("corrected").unwrap().as_str().unwrap(), "the");
    }

    #[test]
    fn test_analytics_event_mode_changed() {
        let e = AnalyticsEvent::mode_changed("Slack", "casual");
        assert_eq!(e.event_type, EventType::ModeChanged);
        let props = e.properties.as_object().unwrap();
        assert_eq!(props.get("app").unwrap().as_str().unwrap(), "Slack");
        assert_eq!(props.get("mode").unwrap().as_str().unwrap(), "casual");
    }

    #[test]
    fn test_analytics_event_app_switched() {
        let e = AnalyticsEvent::app_switched("Safari", "Slack");
        assert_eq!(e.event_type, EventType::AppSwitched);
        let props = e.properties.as_object().unwrap();
        assert_eq!(props.get("from_app").unwrap().as_str().unwrap(), "Safari");
        assert_eq!(props.get("to_app").unwrap().as_str().unwrap(), "Slack");
    }

    #[test]
    fn test_event_type_serde() {
        let et = EventType::TranscriptionCompleted;
        let json = serde_json::to_string(&et).unwrap();
        assert_eq!(json, "\"transcription_completed\"");
        let decoded: EventType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, et);
    }

    #[test]
    fn test_analytics_event_serde() {
        let e = AnalyticsEvent::transcription_completed(1000, 10, "test");
        let json = serde_json::to_string(&e).unwrap();
        let decoded: AnalyticsEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.event_type, EventType::TranscriptionCompleted);
    }
}
