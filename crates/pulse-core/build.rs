fn main() {
    // Compile the Swift recording indicator helper on macOS.
    #[cfg(target_os = "macos")]
    {
        let swift_src = "swift/indicator.swift";
        let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
        let out_dir = format!("target/{}", profile);

        // Only rebuild when the Swift source changes.
        println!("cargo:rerun-if-changed={}", swift_src);

        let status = std::process::Command::new("swiftc")
            .args([
                swift_src,
                "-o",
                &format!("{}/pulse-indicator", out_dir),
                "-framework",
                "AppKit",
                "-O",
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                eprintln!("Built pulse-indicator helper successfully.");
            }
            Ok(s) => {
                eprintln!(
                    "Warning: swiftc exited with {}. Recording indicator will not be available.",
                    s
                );
            }
            Err(e) => {
                eprintln!(
                    "Warning: Could not run swiftc ({}). Recording indicator will not be available.",
                    e
                );
            }
        }
    }

    // Compile the WhisperKit CoreML helper on macOS.
    #[cfg(target_os = "macos")]
    {
        let package_path = "swift/WhisperHelper";
        let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
        let out_dir = format!("target/{}", profile);
        let swift_config = if profile == "release" { "release" } else { "debug" };

        println!("cargo:rerun-if-changed={}/Sources/main.swift", package_path);
        println!("cargo:rerun-if-changed={}/Package.swift", package_path);

        let status = std::process::Command::new("swift")
            .args([
                "build",
                "-c",
                swift_config,
                "--package-path",
                package_path,
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                // Copy the built binary to target/{profile}/.
                let src = format!(
                    "{}/.build/{}/pulse-whisper-coreml",
                    package_path, swift_config
                );
                let dst = format!("{}/pulse-whisper-coreml", out_dir);
                match std::fs::copy(&src, &dst) {
                    Ok(_) => eprintln!("Built pulse-whisper-coreml helper successfully."),
                    Err(e) => eprintln!(
                        "Warning: Built pulse-whisper-coreml but failed to copy to {}: {}",
                        dst, e
                    ),
                }
            }
            Ok(s) => {
                eprintln!(
                    "Warning: swift build exited with {}. CoreML whisper will not be available.",
                    s
                );
            }
            Err(e) => {
                eprintln!(
                    "Warning: Could not run swift build ({}). CoreML whisper will not be available.",
                    e
                );
            }
        }
    }
}
