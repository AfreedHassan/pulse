//! Tauri command types and #[tauri::command] handlers.

use std::sync::mpsc;

use parking_lot::Mutex;
use pulse::providers::local_whisper::PulseModel;
use pulse::types::WritingMode;
use serde::{Deserialize, Serialize};

// ── Provider ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    Whisper,
    Moonshine,
    CoreML,
}

impl Provider {
    pub fn label(&self) -> &'static str {
        match self {
            Provider::Whisper => "Whisper",
            Provider::Moonshine => "Moonshine",
            Provider::CoreML => "CoreML",
        }
    }
}

// ── Settings types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingSettings {
    pub writing_mode: WritingMode,
    pub paste_mode: bool,
    pub llm_base_url: String,
    pub llm_api_key: String,
    pub llm_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultSettings {
    pub llm_base_url: String,
    pub llm_api_key_set: bool,
    pub llm_model: String,
    pub paste_mode: bool,
    pub hotkey_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size: String,
}

// ── Worker channel ──────────────────────────────────────────────────

pub enum WorkerCommand {
    LoadModel {
        provider: Provider,
        whisper_model: PulseModel,
    },
    StartRecording {
        settings: RecordingSettings,
    },
    StopRecording,
    Shutdown,
}

pub struct WorkerHandle {
    pub cmd_tx: Mutex<mpsc::Sender<WorkerCommand>>,
}

// ── Helper: parse model string → PulseModel ─────────────────────────

fn parse_model(s: &str) -> PulseModel {
    match s {
        "tiny" | "fast" => PulseModel::Fast,
        "base" | "balanced" => PulseModel::Balanced,
        "small" | "quality" => PulseModel::Quality,
        "medium" => PulseModel::Medium,
        "large" => PulseModel::Large,
        "large-v3-turbo" | "turbo" => PulseModel::Turbo,
        "distil-large-v3" | "distil" => PulseModel::DistilLarge,
        _ => PulseModel::Large,
    }
}

// ── Tauri commands ──────────────────────────────────────────────────

#[tauri::command]
pub fn load_model(
    provider: String,
    whisper_model: String,
    worker: tauri::State<'_, WorkerHandle>,
) -> Result<(), String> {
    let prov = match provider.as_str() {
        "Whisper" => Provider::Whisper,
        "Moonshine" => Provider::Moonshine,
        "CoreML" => Provider::CoreML,
        _ => Provider::CoreML,
    };
    let model = parse_model(&whisper_model);

    worker
        .cmd_tx
        .lock()
        .send(WorkerCommand::LoadModel {
            provider: prov,
            whisper_model: model,
        })
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn start_recording(
    settings: RecordingSettings,
    worker: tauri::State<'_, WorkerHandle>,
) -> Result<(), String> {
    worker
        .cmd_tx
        .lock()
        .send(WorkerCommand::StartRecording { settings })
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn stop_recording(worker: tauri::State<'_, WorkerHandle>) -> Result<(), String> {
    worker
        .cmd_tx
        .lock()
        .send(WorkerCommand::StopRecording)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_default_settings() -> DefaultSettings {
    DefaultSettings {
        llm_base_url: std::env::var("LLM_BASE_URL").unwrap_or_default(),
        llm_api_key_set: std::env::var("LLM_API_KEY").is_ok(),
        llm_model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
        paste_mode: true,
        hotkey_enabled: true,
    }
}

#[tauri::command]
pub fn get_providers() -> Vec<ProviderInfo> {
    vec![
        ProviderInfo {
            id: "Whisper".into(),
            name: "Whisper".into(),
            description: "Pure Rust, Metal GPU".into(),
            models: vec![
                ModelInfo { id: "tiny".into(), name: "Tiny".into(), size: "39M".into() },
                ModelInfo { id: "base".into(), name: "Base".into(), size: "74M".into() },
                ModelInfo { id: "small".into(), name: "Small".into(), size: "244M".into() },
                ModelInfo { id: "medium".into(), name: "Medium".into(), size: "769M".into() },
                ModelInfo { id: "large".into(), name: "Large".into(), size: "1.5G".into() },
                ModelInfo { id: "distil-large-v3".into(), name: "Distil Large v3".into(), size: "1.5G".into() },
                ModelInfo { id: "large-v3-turbo".into(), name: "Turbo".into(), size: "1.6G".into() },
            ],
        },
        ProviderInfo {
            id: "Moonshine".into(),
            name: "Moonshine".into(),
            description: "ONNX Runtime, CoreML".into(),
            models: vec![
                ModelInfo { id: "moonshine".into(), name: "Moonshine".into(), size: "auto".into() },
            ],
        },
        ProviderInfo {
            id: "CoreML".into(),
            name: "CoreML".into(),
            description: "WhisperKit, ANE".into(),
            models: vec![
                ModelInfo { id: "turbo".into(), name: "Turbo".into(), size: "auto".into() },
            ],
        },
    ]
}
