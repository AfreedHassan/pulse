import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { RecordingSettings, DefaultSettings, ProviderInfo } from "./types";

export async function loadModel(provider: string, whisperModel: string): Promise<void> {
  return invoke("load_model", { provider, whisperModel });
}

export async function startRecording(settings: RecordingSettings): Promise<void> {
  return invoke("start_recording", { settings });
}

export async function stopRecording(): Promise<void> {
  return invoke("stop_recording");
}

export async function getDefaultSettings(): Promise<DefaultSettings> {
  return invoke("get_default_settings");
}

export async function getProviders(): Promise<ProviderInfo[]> {
  return invoke("get_providers");
}

export function onModelLoading(cb: (msg: string) => void): Promise<UnlistenFn> {
  return listen<string>("pulse:model-loading", (e) => cb(e.payload));
}

export function onModelReady(cb: () => void): Promise<UnlistenFn> {
  return listen("pulse:model-ready", () => cb());
}

export function onRecordingStarted(cb: () => void): Promise<UnlistenFn> {
  return listen("pulse:recording-started", () => cb());
}

export function onAudioLevel(cb: (level: number) => void): Promise<UnlistenFn> {
  return listen<number>("pulse:audio-level", (e) => cb(e.payload));
}

export function onChunk(cb: (text: string) => void): Promise<UnlistenFn> {
  return listen<string>("pulse:chunk", (e) => cb(e.payload));
}

export function onDone(cb: (data: { text: string; duration_secs: number }) => void): Promise<UnlistenFn> {
  return listen("pulse:done", (e) => cb(e.payload as { text: string; duration_secs: number }));
}

export function onError(cb: (msg: string) => void): Promise<UnlistenFn> {
  return listen<string>("pulse:error", (e) => cb(e.payload));
}

export function onHotkeyStart(cb: () => void): Promise<UnlistenFn> {
  return listen("pulse:hotkey-start", () => cb());
}

export function onHotkeyStop(cb: () => void): Promise<UnlistenFn> {
  return listen("pulse:hotkey-stop", () => cb());
}
