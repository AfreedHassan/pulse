//! SQLite storage layer for persisting transcriptions, shortcuts, corrections, and settings.

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use tracing::{debug, info};
use uuid::Uuid;

use super::migrations;
use crate::error::Result;
use crate::types::*;

/// Storage backend using SQLite.
pub struct Storage {
    conn: Mutex<Connection>,
}

impl Storage {
    /// Open or create a database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.init()?;
        Ok(storage)
    }

    /// Create an in-memory database (useful for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.init()?;
        Ok(storage)
    }

    fn init(&self) -> Result<()> {
        let conn = self.conn.lock();

        match migrations::run_migrations(&conn) {
            Ok(count) => {
                if count > 0 {
                    info!("Applied {} database migration(s)", count);
                }
            }
            Err(e) => {
                return Err(crate::error::Error::Storage(e));
            }
        }

        // Seed default corrections if table is empty.
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM corrections", [], |row| row.get(0))?;

        if count == 0 {
            let now = Utc::now().to_rfc3339();
            let seeds = [
                ("gonna", "going to"),
                ("wanna", "want to"),
                ("kinda", "kind of"),
                ("gotta", "got to"),
                ("lemme", "let me"),
            ];
            for (original, corrected) in seeds {
                conn.execute(
                    r#"
                    INSERT OR IGNORE INTO corrections (id, original, corrected, occurrences, confidence, source, created_at, updated_at)
                    VALUES (?1, ?2, ?3, 3, 0.75, 'seeded', ?4, ?4)
                    "#,
                    params![Uuid::new_v4().to_string(), original, corrected, now],
                )?;
            }
            debug!("Seeded {} default corrections", seeds.len());
        }

        info!("Database initialized");
        Ok(())
    }

    // ── Transcriptions ──────────────────────────────────────────

    /// Save a transcription.
    pub fn save_transcription(&self, t: &Transcription) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO transcriptions (id, raw_text, processed_text, confidence, duration_ms,
                                        app_name, bundle_id, window_title, app_category, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                t.id.to_string(),
                t.raw_text,
                t.processed_text,
                t.confidence,
                t.duration_ms as i64,
                t.app_context.as_ref().map(|c| &c.app_name),
                t.app_context.as_ref().and_then(|c| c.bundle_id.as_ref()),
                t.app_context.as_ref().and_then(|c| c.window_title.as_ref()),
                t.app_context.as_ref().map(|c| format!("{:?}", c.category)),
                t.created_at.to_rfc3339(),
            ],
        )?;
        debug!("Saved transcription {}", t.id);
        Ok(())
    }

    /// Get recent transcriptions.
    pub fn get_recent_transcriptions(&self, limit: usize) -> Result<Vec<Transcription>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            r#"
            SELECT id, raw_text, processed_text, confidence, duration_ms,
                   app_name, bundle_id, window_title, app_category, created_at
            FROM transcriptions
            ORDER BY created_at DESC
            LIMIT ?1
            "#,
        )?;

        let rows = stmt
            .query_map([limit as i64], |row| {
                let id: String = row.get(0)?;
                let app_name: Option<String> = row.get(5)?;
                let bundle_id: Option<String> = row.get(6)?;
                let window_title: Option<String> = row.get(7)?;
                let _app_category_str: Option<String> = row.get(8)?;
                let created_at_str: String = row.get(9)?;

                let app_context = app_name.map(|name| AppContext {
                    category: AppCategory::from_app(&name, bundle_id.as_deref()),
                    app_name: name,
                    bundle_id,
                    window_title,
                });

                Ok(Transcription {
                    id: Uuid::parse_str(&id).unwrap_or_else(|_| Uuid::new_v4()),
                    raw_text: row.get(1)?,
                    processed_text: row.get(2)?,
                    confidence: row.get(3)?,
                    duration_ms: row.get::<_, i64>(4)? as u64,
                    app_context,
                    created_at: DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// Save a transcription history entry.
    pub fn save_history_entry(&self, entry: &TranscriptionHistoryEntry) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO transcription_history (id, status, text, raw_text, error, duration_ms,
                                               app_name, bundle_id, window_title, app_category, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            "#,
            params![
                entry.id.to_string(),
                match entry.status {
                    TranscriptionStatus::Success => "success",
                    TranscriptionStatus::Failed => "failed",
                },
                entry.text,
                entry.raw_text,
                entry.error,
                entry.duration_ms as i64,
                entry.app_context.as_ref().map(|c| &c.app_name),
                entry.app_context.as_ref().and_then(|c| c.bundle_id.as_ref()),
                entry.app_context.as_ref().and_then(|c| c.window_title.as_ref()),
                entry.app_context.as_ref().map(|c| format!("{:?}", c.category)),
                entry.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    // ── Settings ────────────────────────────────────────────────

    /// Save or update a setting.
    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO settings (key, value, updated_at)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            "#,
            params![key, value, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Get a setting value.
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT value FROM settings WHERE key = ?1",
            params![key],
            |row| row.get(0),
        )
        .optional()
        .map_err(Into::into)
    }

    // ── Shortcuts ───────────────────────────────────────────────

    /// Save a shortcut.
    pub fn save_shortcut(&self, s: &Shortcut) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT OR REPLACE INTO shortcuts (id, trigger, replacement, case_sensitive, enabled, use_count, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            params![
                s.id.to_string(),
                s.trigger,
                s.replacement,
                s.case_sensitive as i32,
                s.enabled as i32,
                s.use_count,
                s.created_at.to_rfc3339(),
                s.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Get all enabled shortcuts.
    pub fn get_enabled_shortcuts(&self) -> Result<Vec<Shortcut>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, trigger, replacement, case_sensitive, enabled, use_count, created_at, updated_at
             FROM shortcuts WHERE enabled = 1 ORDER BY trigger",
        )?;

        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let created_at_str: String = row.get(6)?;
                let updated_at_str: String = row.get(7)?;
                Ok(Shortcut {
                    id: Uuid::parse_str(&id).unwrap_or_else(|_| Uuid::new_v4()),
                    trigger: row.get(1)?,
                    replacement: row.get(2)?,
                    case_sensitive: row.get::<_, i32>(3)? != 0,
                    enabled: row.get::<_, i32>(4)? != 0,
                    use_count: row.get::<_, u32>(5)?,
                    created_at: DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// Delete a shortcut by ID.
    pub fn delete_shortcut(&self, id: &Uuid) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM shortcuts WHERE id = ?1",
            params![id.to_string()],
        )?;
        Ok(())
    }

    // ── Corrections ─────────────────────────────────────────────

    /// Save or update a correction (increments occurrences if already exists).
    pub fn save_correction(&self, c: &Correction) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();

        // Try to update existing.
        let updated = conn.execute(
            r#"
            UPDATE corrections
            SET occurrences = occurrences + 1, updated_at = ?1
            WHERE original = ?2 AND corrected = ?3
            "#,
            params![now, c.original, c.corrected],
        )?;

        if updated == 0 {
            conn.execute(
                r#"
                INSERT INTO corrections (id, original, corrected, occurrences, confidence, source, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
                "#,
                params![
                    c.id.to_string(),
                    c.original,
                    c.corrected,
                    c.occurrences,
                    c.confidence,
                    format!("{:?}", c.source).to_lowercase(),
                    now,
                ],
            )?;
        }
        Ok(())
    }

    /// Get corrections above a confidence threshold.
    pub fn get_corrections(&self, min_confidence: f32) -> Result<Vec<Correction>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, original, corrected, occurrences, confidence, source, created_at, updated_at
             FROM corrections WHERE confidence >= ?1 ORDER BY confidence DESC",
        )?;

        let rows = stmt
            .query_map([min_confidence as f64], |row| {
                let id: String = row.get(0)?;
                let source_str: String = row.get(5)?;
                let created_at_str: String = row.get(6)?;
                let updated_at_str: String = row.get(7)?;

                let source = match source_str.as_str() {
                    "useredit" | "user_edit" => CorrectionSource::UserEdit,
                    "clipboarddiff" | "clipboard_diff" => CorrectionSource::ClipboardDiff,
                    "imported" => CorrectionSource::Imported,
                    _ => CorrectionSource::Seeded,
                };

                Ok(Correction {
                    id: Uuid::parse_str(&id).unwrap_or_else(|_| Uuid::new_v4()),
                    original: row.get(1)?,
                    corrected: row.get(2)?,
                    occurrences: row.get::<_, u32>(3)?,
                    confidence: row.get(4)?,
                    source,
                    created_at: DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    // ── App modes ───────────────────────────────────────────────

    /// Save per-app writing mode.
    pub fn save_app_mode(&self, app_name: &str, mode: WritingMode) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO app_modes (app_name, mode, updated_at)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(app_name) DO UPDATE SET
                mode = excluded.mode,
                updated_at = excluded.updated_at
            "#,
            params![
                app_name,
                format!("{:?}", mode).to_lowercase(),
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Get per-app writing mode.
    pub fn get_app_mode(&self, app_name: &str) -> Result<Option<WritingMode>> {
        let conn = self.conn.lock();
        let mode_str: Option<String> = conn
            .query_row(
                "SELECT mode FROM app_modes WHERE app_name = ?1",
                params![app_name],
                |row| row.get(0),
            )
            .optional()?;

        Ok(mode_str.and_then(|s| WritingMode::parse(&s)))
    }

    // ── Analytics ───────────────────────────────────────────────

    /// Save an analytics event.
    pub fn save_event(&self, event: &AnalyticsEvent) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO analytics_events (id, event_type, properties, app_name, bundle_id, window_title, app_category, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            params![
                event.id.to_string(),
                format!("{:?}", event.event_type).to_lowercase(),
                event.properties.to_string(),
                event.app_context.as_ref().map(|c| &c.app_name),
                event.app_context.as_ref().and_then(|c| c.bundle_id.as_ref()),
                event.app_context.as_ref().and_then(|c| c.window_title.as_ref()),
                event.app_context.as_ref().map(|c| format!("{:?}", c.category)),
                event.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    // ── Contacts ────────────────────────────────────────────────

    /// Save a contact.
    pub fn save_contact(&self, contact: &Contact) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT OR REPLACE INTO contacts (id, name, category, created_at)
            VALUES (?1, ?2, ?3, ?4)
            "#,
            params![
                contact.id.to_string(),
                contact.name,
                format!("{:?}", contact.category).to_lowercase(),
                contact.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Get all contacts.
    pub fn get_contacts(&self) -> Result<Vec<Contact>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT id, name, category, created_at FROM contacts ORDER BY name")?;

        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let category_str: String = row.get(2)?;
                let created_at_str: String = row.get(3)?;

                let category = match category_str.as_str() {
                    "close" => ContactCategory::Close,
                    "professional" => ContactCategory::Professional,
                    _ => ContactCategory::Acquaintance,
                };

                Ok(Contact {
                    id: Uuid::parse_str(&id).unwrap_or_else(|_| Uuid::new_v4()),
                    name: row.get(1)?,
                    category,
                    created_at: DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_storage() -> Storage {
        Storage::in_memory().expect("Failed to create in-memory storage")
    }

    #[test]
    fn test_open_and_init() {
        let storage = test_storage();
        // Seeded corrections should exist.
        let corrections = storage.get_corrections(0.0).unwrap();
        assert!(!corrections.is_empty(), "Should have seeded corrections");
    }

    #[test]
    fn test_settings_crud() {
        let storage = test_storage();
        assert_eq!(storage.get_setting("key1").unwrap(), None);

        storage.set_setting("key1", "value1").unwrap();
        assert_eq!(storage.get_setting("key1").unwrap(), Some("value1".into()));

        storage.set_setting("key1", "value2").unwrap();
        assert_eq!(storage.get_setting("key1").unwrap(), Some("value2".into()));
    }

    #[test]
    fn test_transcription_save_and_retrieve() {
        let storage = test_storage();

        let t = Transcription::new("hello world".into(), "Hello, world.".into(), 0.95, 1200);
        storage.save_transcription(&t).unwrap();

        let recent = storage.get_recent_transcriptions(10).unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].raw_text, "hello world");
        assert_eq!(recent[0].processed_text, "Hello, world.");
    }

    #[test]
    fn test_shortcuts_crud() {
        let storage = test_storage();

        let s = Shortcut::new("my email".into(), "user@example.com".into());
        storage.save_shortcut(&s).unwrap();

        let shortcuts = storage.get_enabled_shortcuts().unwrap();
        assert_eq!(shortcuts.len(), 1);
        assert_eq!(shortcuts[0].trigger, "my email");

        storage.delete_shortcut(&s.id).unwrap();
        let shortcuts = storage.get_enabled_shortcuts().unwrap();
        assert!(shortcuts.is_empty());
    }

    #[test]
    fn test_correction_save_and_increment() {
        let storage = test_storage();

        let c = Correction::new("teh".into(), "the".into(), CorrectionSource::UserEdit);
        storage.save_correction(&c).unwrap();

        // Saving again should increment occurrences, not add a new row.
        storage.save_correction(&c).unwrap();

        let corrections = storage.get_corrections(0.0).unwrap();
        let found = corrections.iter().find(|x| x.original == "teh");
        assert!(found.is_some());
        assert!(found.unwrap().occurrences >= 2);
    }

    #[test]
    fn test_app_modes() {
        let storage = test_storage();

        assert_eq!(storage.get_app_mode("Slack").unwrap(), None);

        storage.save_app_mode("Slack", WritingMode::Casual).unwrap();
        assert_eq!(
            storage.get_app_mode("Slack").unwrap(),
            Some(WritingMode::Casual)
        );

        storage
            .save_app_mode("Slack", WritingMode::VeryCasual)
            .unwrap();
        assert_eq!(
            storage.get_app_mode("Slack").unwrap(),
            Some(WritingMode::VeryCasual)
        );
    }

    #[test]
    fn test_history_entry() {
        let storage = test_storage();

        let entry = TranscriptionHistoryEntry::success("raw".into(), "processed".into(), 500);
        storage.save_history_entry(&entry).unwrap();

        let fail_entry = TranscriptionHistoryEntry::failure("oops".into(), 100);
        storage.save_history_entry(&fail_entry).unwrap();
    }

    #[test]
    fn test_contacts() {
        let storage = test_storage();

        let c = Contact::new("Alice".into(), ContactCategory::Close);
        storage.save_contact(&c).unwrap();

        let contacts = storage.get_contacts().unwrap();
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].name, "Alice");
    }
}
