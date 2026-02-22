//! Database schema migrations for Pulse.
//!
//! Each migration is a SQL string applied sequentially. The schema_version
//! table tracks which migrations have already been applied.

use rusqlite::Connection;

/// All migrations in order. Each entry is (version, description, sql).
const MIGRATIONS: &[(i32, &str, &str)] = &[(
    1,
    "Initial schema",
    r#"
        CREATE TABLE IF NOT EXISTS transcriptions (
            id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL,
            processed_text TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            app_name TEXT,
            bundle_id TEXT,
            window_title TEXT,
            app_category TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS transcription_history (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'success',
            text TEXT NOT NULL DEFAULT '',
            raw_text TEXT NOT NULL DEFAULT '',
            error TEXT,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            app_name TEXT,
            bundle_id TEXT,
            window_title TEXT,
            app_category TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS shortcuts (
            id TEXT PRIMARY KEY,
            trigger TEXT NOT NULL,
            replacement TEXT NOT NULL,
            case_sensitive INTEGER NOT NULL DEFAULT 0,
            enabled INTEGER NOT NULL DEFAULT 1,
            use_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_shortcuts_trigger ON shortcuts(trigger);

        CREATE TABLE IF NOT EXISTS corrections (
            id TEXT PRIMARY KEY,
            original TEXT NOT NULL,
            corrected TEXT NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            confidence REAL NOT NULL DEFAULT 0.5,
            source TEXT NOT NULL DEFAULT 'user_edit',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_corrections_original ON corrections(original);

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS app_modes (
            app_name TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analytics_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            properties TEXT NOT NULL DEFAULT '{}',
            app_name TEXT,
            bundle_id TEXT,
            window_title TEXT,
            app_category TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_type ON analytics_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_created ON analytics_events(created_at);

        CREATE TABLE IF NOT EXISTS contacts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'acquaintance',
            created_at TEXT NOT NULL
        );
        "#,
)];

/// Run all pending migrations. Returns the number of migrations applied.
pub fn run_migrations(conn: &Connection) -> std::result::Result<usize, rusqlite::Error> {
    // Ensure the schema_version table exists.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );",
    )?;

    let current_version: i32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let mut applied = 0;
    for &(version, description, sql) in MIGRATIONS {
        if version > current_version {
            conn.execute_batch(sql)?;
            conn.execute(
                "INSERT INTO schema_version (version, description, applied_at) VALUES (?1, ?2, ?3)",
                rusqlite::params![version, description, chrono::Utc::now().to_rfc3339()],
            )?;
            applied += 1;
        }
    }

    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_migrations_fresh_db() {
        let conn = Connection::open_in_memory().unwrap();
        let applied = run_migrations(&conn).unwrap();
        assert_eq!(applied, MIGRATIONS.len());

        // Running again should apply nothing.
        let applied2 = run_migrations(&conn).unwrap();
        assert_eq!(applied2, 0);
    }

    #[test]
    fn test_tables_created() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        let tables = [
            "transcriptions",
            "transcription_history",
            "shortcuts",
            "corrections",
            "settings",
            "app_modes",
            "analytics_events",
            "contacts",
        ];
        for table in tables {
            let count: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM {}", table), [], |row| {
                    row.get(0)
                })
                .unwrap_or_else(|_| panic!("Table '{}' should exist", table));
            assert_eq!(count, 0, "Table '{}' should be empty", table);
        }

        // schema_version should have one row after migrations
        let version_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM schema_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version_count, 1, "schema_version should have one row");
    }

    #[test]
    fn test_schema_version_recorded() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        let version: i32 = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(version, 1);

        let description: String = conn
            .query_row(
                "SELECT description FROM schema_version WHERE version = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(description, "Initial schema");
    }

    #[test]
    fn test_indexes_created() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        let indexes = [
            "idx_shortcuts_trigger",
            "idx_corrections_original",
            "idx_events_type",
            "idx_events_created",
        ];
        for idx in indexes {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?",
                    [idx],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 1, "Index '{}' should exist", idx);
        }
    }

    #[test]
    fn test_migrations_constant() {
        assert!(!MIGRATIONS.is_empty());
        for (i, (version, desc, sql)) in MIGRATIONS.iter().enumerate() {
            assert!(*version > 0, "Version should be positive");
            assert!(!desc.is_empty(), "Description should not be empty");
            assert!(!sql.is_empty(), "SQL should not be empty");
            if i > 0 {
                assert!(
                    *version > MIGRATIONS[i - 1].0,
                    "Versions should be increasing"
                );
            }
        }
    }
}
