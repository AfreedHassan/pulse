//! Correction types for learned text corrections.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type CorrectionId = Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionSource {
    UserEdit,
    ClipboardDiff,
    Imported,
    Seeded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    pub id: CorrectionId,
    pub original: String,
    pub corrected: String,
    pub occurrences: u32,
    pub confidence: f32,
    pub source: CorrectionSource,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Correction {
    pub fn new(original: String, corrected: String, source: CorrectionSource) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            original,
            corrected,
            occurrences: 1,
            confidence: 0.5,
            source,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn update_confidence(&mut self) {
        let e = std::f32::consts::E;
        self.confidence = 0.5 + 0.5 * (1.0 - 1.0 / (self.occurrences as f32 + e).ln());
        self.confidence = self.confidence.min(0.99);
    }

    pub fn increment(&mut self) {
        self.occurrences += 1;
        self.update_confidence();
        self.updated_at = Utc::now();
    }

    pub fn is_applicable(&self, min_confidence: f32) -> bool {
        self.confidence >= min_confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correction_new() {
        let c = Correction::new("teh".into(), "the".into(), CorrectionSource::UserEdit);
        assert_eq!(c.original, "teh");
        assert_eq!(c.corrected, "the");
        assert_eq!(c.occurrences, 1);
        assert_eq!(c.confidence, 0.5);
        assert_eq!(c.source, CorrectionSource::UserEdit);
        assert!(!c.id.is_nil());
    }

    #[test]
    fn test_correction_confidence_scaling() {
        let mut c = Correction::new("teh".into(), "the".into(), CorrectionSource::UserEdit);
        assert_eq!(c.confidence, 0.5);

        c.occurrences = 3;
        c.update_confidence();
        assert!(c.confidence > 0.5);
        assert!(c.confidence < 0.99);

        c.occurrences = 100;
        c.update_confidence();
        assert!(c.confidence > 0.85);
        assert!(c.confidence <= 0.99);
    }

    #[test]
    fn test_correction_confidence_caps_at_99() {
        let mut c = Correction::new("test".into(), "correct".into(), CorrectionSource::Seeded);
        c.occurrences = 1_000_000;
        c.update_confidence();
        assert!(c.confidence <= 0.99);
    }

    #[test]
    fn test_correction_increment() {
        let mut c = Correction::new("teh".into(), "the".into(), CorrectionSource::UserEdit);
        let initial_confidence = c.confidence;
        c.increment();
        assert_eq!(c.occurrences, 2);
        assert!(c.confidence > initial_confidence);
    }

    #[test]
    fn test_correction_is_applicable() {
        let mut c = Correction::new("teh".into(), "the".into(), CorrectionSource::UserEdit);
        assert!(c.is_applicable(0.3));
        assert!(c.is_applicable(0.5));
        assert!(!c.is_applicable(0.8));

        c.occurrences = 100;
        c.update_confidence();
        assert!(c.is_applicable(0.8));
    }

    #[test]
    fn test_correction_source_serde() {
        let sources = [
            CorrectionSource::UserEdit,
            CorrectionSource::ClipboardDiff,
            CorrectionSource::Imported,
            CorrectionSource::Seeded,
        ];
        for source in sources {
            let json = serde_json::to_string(&source).unwrap();
            let decoded: CorrectionSource = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, source);
        }
    }

    #[test]
    fn test_correction_serde() {
        let c = Correction::new(
            "gonna".into(),
            "going to".into(),
            CorrectionSource::UserEdit,
        );
        let json = serde_json::to_string(&c).unwrap();
        let decoded: Correction = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original, "gonna");
        assert_eq!(decoded.corrected, "going to");
    }
}
