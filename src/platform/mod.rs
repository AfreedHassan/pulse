//! macOS-specific platform integration.

pub mod apps;
pub mod hotkey;
pub mod paste;

pub use apps::detect_frontmost_app;
pub use hotkey::listen;
pub use paste::paste_text;
