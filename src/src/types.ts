export type Provider = "Whisper" | "Moonshine" | "CoreML";
export type WritingMode = "formal" | "casual" | "very_casual" | "excited";

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
}

export interface ProviderInfo {
  id: string;
  name: string;
  description: string;
  models: ModelInfo[];
}

export interface RecordingSettings {
  writing_mode: WritingMode;
  paste_mode: boolean;
  llm_base_url: string;
  llm_api_key: string;
  llm_model: string;
}

export interface DefaultSettings {
  llm_base_url: string;
  llm_api_key_set: boolean;
  llm_model: string;
  paste_mode: boolean;
  hotkey_enabled: boolean;
}

export interface TranscriptEntry {
  text: string;
  timestamp: string;
  duration_secs: number;
}

export interface TranscriptionDone {
  text: string;
  duration_secs: number;
}

export type AppPanel = "transcript" | "settings" | "style" | "help";

export interface AppState {
  provider: Provider;
  whisperModel: string;
  modelReady: boolean;
  modelStatus: string;
  recording: boolean;
  recordStart: number | null;
  audioLevel: number;
  liveText: string;
  history: TranscriptEntry[];
  writingMode: WritingMode;
  pasteMode: boolean;
  hotkeyEnabled: boolean;
  llmBaseUrl: string;
  llmApiKey: string;
  llmModel: string;
  activePanel: AppPanel;
  statusText: string;
}
