//! Visual recording indicator (floating pill overlay).
//!
//! Spawns a small Swift helper process that displays a floating pill
//! at the bottom of the screen during recording. Communication is via
//! newline-delimited commands over stdin.

use std::io::Write;
use std::process::{Child, Command, Stdio};

/// Handle to the indicator helper process.
pub struct Indicator {
    child: Child,
}

impl Indicator {
    /// Spawn the indicator helper process.
    ///
    /// Returns `None` if the helper binary is not found (e.g. not on macOS,
    /// or the Swift helper was not compiled).
    pub fn spawn() -> Option<Self> {
        let binary = Self::helper_path()?;

        let child = Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        Some(Self { child })
    }

    /// Show the recording indicator.
    pub fn show(&mut self) {
        self.send("show\n");
    }

    /// Hide the recording indicator.
    pub fn hide(&mut self) {
        self.send("hide\n");
    }

    /// Update the audio level (0.0–1.0) for waveform animation.
    pub fn set_level(&mut self, level: f32) {
        self.send(&format!("level {:.2}\n", level.clamp(0.0, 1.0)));
    }

    fn send(&mut self, msg: &str) {
        if let Some(stdin) = self.child.stdin.as_mut() {
            let _ = stdin.write_all(msg.as_bytes());
            let _ = stdin.flush();
        }
    }

    /// Locate the helper binary next to the current executable.
    fn helper_path() -> Option<std::path::PathBuf> {
        let exe = std::env::current_exe().ok()?;
        let dir = exe.parent()?;
        let path = dir.join("pulse-indicator");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }
}

impl Drop for Indicator {
    fn drop(&mut self) {
        // Send quit command, then kill if needed.
        self.send("quit\n");
        let _ = self.child.wait();
    }
}
