//! macOS-specific platform integration.

pub mod accessibility;
pub mod apps;
pub mod hotkey;
pub mod indicator;
pub mod paste;

pub use accessibility::read_focused_text_field;
pub use apps::detect_frontmost_app;
pub use hotkey::listen;
pub use paste::paste_text;
