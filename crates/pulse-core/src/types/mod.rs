//! Core domain types for Pulse.

pub mod analytics;
pub mod app;
pub mod contacts;
pub mod corrections;
pub mod modes;
pub mod shortcuts;
pub mod transcription;

pub use analytics::{AnalyticsEvent, EventType};
pub use app::{AppCategory, AppContext};
pub use contacts::{Contact, ContactCategory};
pub use corrections::{Correction, CorrectionId, CorrectionSource};
pub use modes::WritingMode;
pub use shortcuts::{Shortcut, ShortcutId};
pub use transcription::{
    AudioData, Transcription, TranscriptionHistoryEntry, TranscriptionId, TranscriptionStatus,
};
