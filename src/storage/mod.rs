//! Persistence layer for transcriptions, shortcuts, corrections, and settings.

pub mod db;
pub mod migrations;

pub use db::Storage;
pub use migrations::run_migrations;
