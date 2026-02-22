//! Background worker thread: model loading, recording, transcription, and post-processing.
//!
//! Ported from `src/bin/pulse_gui/worker.rs` for Tauri.
//! Emits events via `AppHandle::emit` instead of channel-based `WorkerEvent`.

use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Emitter};

use pulse::audio::AudioCapture;
use pulse::engine::formatter::{self, Formatter, FormatterConfig};
use pulse::engine::{ContactEngine, LearningEngine, ShortcutEngine};
use pulse::platform::indicator::Indicator;
use pulse::platform::paste;
use pulse::providers::coreml_whisper::{CoreMLModel, CoreMLWhisperEngine};
use pulse::providers::local_whisper::PulseEngine;
use pulse::providers::moonshine::MoonshineEngine;
use pulse::providers::Engine;
use pulse::storage::Storage;

use crate::commands::{Provider, RecordingSettings, WorkerCommand};

/// How often to poll audio level and check for stop during recording.
const POLL_INTERVAL: Duration = Duration::from_millis(32);

/// Minimum samples before dispatching a chunk (0.5s at 16kHz).
const MIN_CHUNK_SAMPLES: usize = 8_000;

/// How often to drain the audio buffer and dispatch for transcription.
const CHUNK_INTERVAL: Duration = Duration::from_secs(5);

/// Emit a Tauri event, logging failures to stderr.
fn emit<S: serde::Serialize + Clone>(app: &AppHandle, event: &str, payload: S) {
    if let Err(e) = app.emit(event, payload) {
        eprintln!("[tauri] Failed to emit {}: {}", event, e);
    }
}

pub fn spawn(cmd_rx: Receiver<WorkerCommand>, app_handle: AppHandle) {
    thread::spawn(move || run(cmd_rx, app_handle));
}

fn open_storage() -> Option<Storage> {
    let db_path = pulse::data_dir().join("pulse.db");
    std::fs::create_dir_all(db_path.parent().unwrap()).ok();
    Storage::open(&db_path).ok()
}

/// Apply the full post-transcription pipeline.
fn process_text(raw: &str, settings: &RecordingSettings, storage: Option<&Storage>) -> String {
    let mut text = raw.to_string();

    if let Some(storage) = storage {
        let learning = LearningEngine::new(storage);
        text = learning.apply(&text);

        let shortcuts = ShortcutEngine::new(storage);
        text = shortcuts.expand(&text);

        let contacts = ContactEngine::new(storage);
        text = contacts.apply_names(&text);
    }

    // Use env vars as fallback so the GUI works even when the user only
    // configured LLM settings via environment (matching CLI behaviour).
    let llm_base_url = if settings.llm_base_url.is_empty() {
        std::env::var("LLM_BASE_URL").unwrap_or_default()
    } else {
        settings.llm_base_url.clone()
    };

    if !llm_base_url.is_empty() {
        let api_key = if settings.llm_api_key.is_empty() {
            std::env::var("LLM_API_KEY").unwrap_or_default()
        } else {
            settings.llm_api_key.clone()
        };
        let model = if settings.llm_model.is_empty() {
            std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into())
        } else {
            settings.llm_model.clone()
        };
        let cfg = FormatterConfig {
            base_url: llm_base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
        };
        let fmt = Formatter::new(cfg);
        let mode = settings.writing_mode.clone();
        match fmt.format_with_mode(&text, Some(&mode), None) {
            Ok(formatted) if formatter::passes_guardrails(&text, &formatted) => {
                text = formatted;
            }
            Ok(_) => {
                eprintln!("[tauri] Formatter produced bad output, using raw text");
            }
            Err(e) => {
                eprintln!("[tauri] Formatter error: {}", e);
            }
        }
    }

    text
}

fn run(cmd_rx: Receiver<WorkerCommand>, app: AppHandle) {
    let mut engine: Option<Engine> = None;
    let storage = open_storage();
    let mut indicator = Indicator::spawn();
    if indicator.is_some() {
        eprintln!("[tauri] Recording indicator ready");
    }

    loop {
        let cmd = match cmd_rx.recv() {
            Ok(c) => c,
            Err(_) => break,
        };

        match cmd {
            WorkerCommand::LoadModel {
                provider,
                whisper_model,
            } => {
                emit(
                    &app,
                    "pulse:model-loading",
                    format!("Loading {} model...", provider.label()),
                );

                engine = None;

                let result = match provider {
                    Provider::Whisper => PulseEngine::new(whisper_model)
                        .map(Engine::Whisper)
                        .map_err(|e| e.to_string()),
                    Provider::Moonshine => MoonshineEngine::new()
                        .map(Engine::Moonshine)
                        .map_err(|e| e.to_string()),
                    Provider::CoreML => CoreMLWhisperEngine::new(CoreMLModel::Large)
                        .map(Engine::CoreML)
                        .map_err(|e| e.to_string()),
                };

                match result {
                    Ok(eng) => {
                        engine = Some(eng);
                        emit(&app, "pulse:model-ready", ());
                    }
                    Err(e) => {
                        emit(&app, "pulse:model-error", e);
                    }
                }
            }

            WorkerCommand::StartRecording { settings } => {
                let cap = AudioCapture::new();
                cap.start();
                emit(&app, "pulse:recording-started", ());

                if let Some(ref mut ind) = indicator {
                    ind.show();
                }

                let sample_rate = cap.sample_rate();
                let channels = cap.channels();
                let rate_scale = (sample_rate as usize / 16_000).max(1);
                let min_chunk = MIN_CHUNK_SAMPLES * rate_scale;

                let mut chunks: Vec<String> = Vec::new();
                let mut last_drain = Instant::now();
                let record_start = Instant::now();

                loop {
                    if let Ok(WorkerCommand::StopRecording) = cmd_rx.try_recv() {
                        break;
                    }

                    // Audio level.
                    let live_len = cap.live_len();
                    if live_len > 0 {
                        let tail = cap.peek_tail(1600);
                        let rms =
                            (tail.iter().map(|s| s * s).sum::<f32>() / tail.len() as f32).sqrt();
                        emit(&app, "pulse:audio-level", rms);
                        if let Some(ref mut ind) = indicator {
                            ind.set_level(rms);
                        }
                    }

                    // Chunk-based transcription.
                    if last_drain.elapsed() >= CHUNK_INTERVAL {
                        let available = cap.live_len();
                        if available > min_chunk {
                            let chunk = cap.drain(available);
                            if let Some(ref mut eng) = engine {
                                match eng.transcribe_sync(&chunk, sample_rate, channels) {
                                    Ok(text) if !text.is_empty() => {
                                        emit(&app, "pulse:chunk", text.clone());
                                        chunks.push(text);
                                    }
                                    Ok(_) => {}
                                    Err(e) => {
                                        emit(&app, "pulse:error", e.to_string());
                                    }
                                }
                            }
                        }
                        last_drain = Instant::now();
                    }

                    thread::sleep(POLL_INTERVAL);
                }

                // Stop and process remaining audio.
                cap.stop();
                if let Some(ref mut ind) = indicator {
                    ind.hide();
                }
                let remaining = cap.samples();

                if !remaining.is_empty() && remaining.len() > min_chunk {
                    if let Some(ref mut eng) = engine {
                        match eng.transcribe_sync(&remaining, sample_rate, channels) {
                            Ok(text) if !text.is_empty() => {
                                chunks.push(text);
                            }
                            Ok(_) => {}
                            Err(e) => {
                                emit(&app, "pulse:error", e.to_string());
                            }
                        }
                    }
                }

                let raw_text = chunks.join(" ");
                let actual_duration = record_start.elapsed().as_secs_f64();

                if raw_text.is_empty() {
                    emit(
                        &app,
                        "pulse:done",
                        serde_json::json!({ "text": "", "duration_secs": actual_duration }),
                    );
                } else {
                    let processed = process_text(&raw_text, &settings, storage.as_ref());

                    if settings.paste_mode {
                        if let Err(e) = paste::paste_text(&processed) {
                            eprintln!("[tauri] Paste failed: {}", e);
                        }
                    }

                    emit(
                        &app,
                        "pulse:done",
                        serde_json::json!({
                            "text": processed,
                            "duration_secs": actual_duration
                        }),
                    );
                }
            }

            WorkerCommand::StopRecording => {
                // No-op if not recording.
            }

            WorkerCommand::Shutdown => {
                break;
            }
        }
    }
}
