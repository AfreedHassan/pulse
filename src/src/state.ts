import type { AppState } from "./types";

export function createState(): AppState {
  return {
    provider: "CoreML",
    whisperModel: "large",
    modelReady: false,
    modelStatus: "Select a model to begin",
    recording: false,
    recordStart: null,
    audioLevel: 0,
    liveText: "",
    history: [],
    writingMode: "casual",
    pasteMode: true,
    hotkeyEnabled: false,
    llmBaseUrl: "",
    llmApiKey: "",
    llmModel: "",
    activePanel: "transcript",
    statusText: "",
  };
}
