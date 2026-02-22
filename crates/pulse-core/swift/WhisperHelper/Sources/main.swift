// Pulse CoreML Whisper helper — persistent subprocess using WhisperKit.
//
// Protocol (newline-delimited over stdin/stdout):
//   "load <model_variant>\n"     → "ok loaded\n"
//   "transcribe <wav_path>\n"    → "ok <transcribed text>\n"
//   Errors:                      → "error <message>\n"
//
// Stderr is inherited for download progress visibility.

import Foundation
import WhisperKit

// Unbuffer stdout so responses are visible to the parent process immediately.
setbuf(stdout, nil)

/// Map CLI model names to WhisperKit model identifiers.
/// These match folder names in the argmaxinc/whisperkit-coreml HuggingFace repo.
/// The `_turbo` suffix denotes WhisperKit's optimized CoreML compilation.
func resolveModelName(_ variant: String) -> String {
    switch variant {
    case "large-v3-turbo", "turbo":
        return "openai_whisper-large-v3-v20240930_turbo"
    case "large-v3", "large":
        return "openai_whisper-large-v3_turbo"
    case "distil-large-v3", "distil":
        return "distil-whisper_distil-large-v3_turbo"
    default:
        return variant
    }
}

func respond(_ message: String) {
    print(message)
    fflush(stdout)
}

var whisperKit: WhisperKit?

while let line = readLine() {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { continue }

    let parts = trimmed.split(separator: " ", maxSplits: 1)
    let command = String(parts[0])

    switch command {
    case "load":
        guard parts.count == 2 else {
            respond("error missing model variant for load command")
            continue
        }
        let variant = String(parts[1])
        let modelName = resolveModelName(variant)

        fputs("WhisperKit: loading model \(modelName)...\n", stderr)

        do {
            // Download model first with progress callback so user sees activity.
            fputs("WhisperKit: downloading model (cached after first run)...\n", stderr)
            var lastPct = -1
            let modelURL = try await WhisperKit.download(
                variant: modelName,
                from: "argmaxinc/whisperkit-coreml",
                progressCallback: { progress in
                    let pct = Int(progress.fractionCompleted * 100)
                    if pct != lastPct && pct % 5 == 0 {
                        lastPct = pct
                        fputs("WhisperKit: download \(pct)%\n", stderr)
                    }
                }
            )
            fputs("WhisperKit: model downloaded, loading...\n", stderr)

            // Load from local folder — no re-download.
            let config = WhisperKitConfig(
                modelFolder: modelURL.path,
                computeOptions: ModelComputeOptions(
                    audioEncoderCompute: .cpuAndNeuralEngine,
                    textDecoderCompute: .cpuAndNeuralEngine
                ),
                verbose: false,
                logLevel: .error,
                load: true,
                download: false
            )
            whisperKit = try await WhisperKit(config)
            fputs("WhisperKit: model loaded successfully\n", stderr)
            respond("ok loaded")
        } catch {
            respond("error failed to load model: \(error.localizedDescription)")
        }

    case "transcribe":
        guard parts.count == 2 else {
            respond("error missing wav path for transcribe command")
            continue
        }
        let wavPath = String(parts[1])

        guard let kit = whisperKit else {
            respond("error model not loaded, send load command first")
            continue
        }

        guard FileManager.default.fileExists(atPath: wavPath) else {
            respond("error file not found: \(wavPath)")
            continue
        }

        do {
            let results = try await kit.transcribe(audioPath: wavPath)
            let text = results
                .map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
                .joined(separator: " ")
                .replacingOccurrences(of: "\n", with: " ")
            respond("ok \(text)")
        } catch {
            respond("error transcription failed: \(error.localizedDescription)")
        }

    default:
        respond("error unknown command: \(command)")
    }
}

// stdin closed — parent process exited
exit(0)
