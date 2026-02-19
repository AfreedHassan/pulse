//! Voice Activity Detection (VAD) module.
//!
//! Provides a simple energy-based VAD with a state machine for tracking
//! speech/silence transitions. This replaces the inline VAD logic
//! previously embedded in main.rs.

/// Voice activity state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceActivity {
    Silence,
    Speech,
}

/// Model's expected sample rate (16kHz).
pub const VAD_SAMPLE_RATE: u32 = 16_000;
/// Number of samples per VAD chunk (~32ms at 16kHz).
pub const VAD_CHUNK_SIZE: usize = 512;

/// Simple energy-based Voice Activity Detector with hysteresis.
///
/// Uses consecutive chunk counting to prevent rapid toggling between
/// speech and silence states. Transitions require a minimum number
/// of consecutive chunks in the new state before switching.
pub struct SimpleVad {
    /// RMS energy threshold above which audio is considered speech.
    threshold: f32,
    /// Number of consecutive speech chunks required to transition from Silence → Speech.
    min_speech_chunks: usize,
    /// Number of consecutive silence chunks required to transition from Speech → Silence.
    min_silence_chunks: usize,
    /// Consecutive speech chunk counter.
    speech_chunk_count: usize,
    /// Consecutive silence chunk counter.
    silence_chunk_count: usize,
    /// Current detected state.
    current_state: VoiceActivity,
}

impl Default for SimpleVad {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleVad {
    /// Create a new VAD with default thresholds.
    ///
    /// Defaults:
    /// - `threshold`: 0.01 RMS (~-40 dB)
    /// - `min_speech_chunks`: 3 (~96ms at VAD_CHUNK_SIZE=512, 16kHz)
    /// - `min_silence_chunks`: 15 (~480ms at VAD_CHUNK_SIZE=512, 16kHz)
    pub fn new() -> Self {
        Self {
            threshold: 0.01,
            min_speech_chunks: 3,
            min_silence_chunks: 15,
            speech_chunk_count: 0,
            silence_chunk_count: 0,
            current_state: VoiceActivity::Silence,
        }
    }

    /// Create a VAD with custom thresholds.
    pub fn with_thresholds(
        threshold: f32,
        min_speech_chunks: usize,
        min_silence_chunks: usize,
    ) -> Self {
        Self {
            threshold,
            min_speech_chunks,
            min_silence_chunks,
            speech_chunk_count: 0,
            silence_chunk_count: 0,
            current_state: VoiceActivity::Silence,
        }
    }

    /// Process an audio chunk and return (current_state, state_just_changed).
    ///
    /// The boolean indicates whether the state transitioned on this call.
    /// This allows the caller to detect utterance boundaries:
    /// - `(Speech, true)` → speech just started
    /// - `(Silence, true)` → speech just ended (utterance boundary)
    pub fn update(&mut self, samples: &[f32]) -> (VoiceActivity, bool) {
        let rms = compute_rms(samples);
        let is_speech = rms > self.threshold;

        match self.current_state {
            VoiceActivity::Silence => {
                if is_speech {
                    self.speech_chunk_count += 1;
                    self.silence_chunk_count = 0;
                    if self.speech_chunk_count >= self.min_speech_chunks {
                        self.current_state = VoiceActivity::Speech;
                        self.speech_chunk_count = 0;
                        return (VoiceActivity::Speech, true);
                    }
                } else {
                    self.speech_chunk_count = 0;
                }
            }
            VoiceActivity::Speech => {
                if !is_speech {
                    self.silence_chunk_count += 1;
                    self.speech_chunk_count = 0;
                    if self.silence_chunk_count >= self.min_silence_chunks {
                        self.current_state = VoiceActivity::Silence;
                        self.silence_chunk_count = 0;
                        return (VoiceActivity::Silence, true);
                    }
                } else {
                    self.silence_chunk_count = 0;
                }
            }
        }

        (self.current_state, false)
    }

    /// Get the current voice activity state.
    pub fn state(&self) -> VoiceActivity {
        self.current_state
    }

    /// Reset the VAD to initial silence state.
    pub fn reset(&mut self) {
        self.current_state = VoiceActivity::Silence;
        self.speech_chunk_count = 0;
        self.silence_chunk_count = 0;
    }

    /// Get the RMS probability from a chunk (0.0 = silence, higher = speech).
    pub fn process_chunk(samples: &[f32]) -> f32 {
        compute_rms(samples)
    }
}

/// Compute Root Mean Square energy.
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_silence(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn make_speech(len: usize, amplitude: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * 0.1).sin() * amplitude)
            .collect()
    }

    #[test]
    fn test_initial_state() {
        let vad = SimpleVad::new();
        assert_eq!(vad.state(), VoiceActivity::Silence);
    }

    #[test]
    fn test_silence_stays_silent() {
        let mut vad = SimpleVad::new();
        let silence = make_silence(VAD_CHUNK_SIZE);

        for _ in 0..20 {
            let (state, changed) = vad.update(&silence);
            assert_eq!(state, VoiceActivity::Silence);
            assert!(!changed);
        }
    }

    #[test]
    fn test_speech_detection_with_hysteresis() {
        let mut vad = SimpleVad::new();
        let speech = make_speech(VAD_CHUNK_SIZE, 0.5);

        // First chunk: not enough consecutive speech chunks yet.
        let (state, changed) = vad.update(&speech);
        assert_eq!(state, VoiceActivity::Silence);
        assert!(!changed);

        // Second chunk: still not enough.
        let (state, changed) = vad.update(&speech);
        assert_eq!(state, VoiceActivity::Silence);
        assert!(!changed);

        // Third chunk: triggers transition.
        let (state, changed) = vad.update(&speech);
        assert_eq!(state, VoiceActivity::Speech);
        assert!(changed);
    }

    #[test]
    fn test_silence_after_speech() {
        let mut vad = SimpleVad::with_thresholds(0.01, 2, 3);
        let speech = make_speech(VAD_CHUNK_SIZE, 0.5);
        let silence = make_silence(VAD_CHUNK_SIZE);

        // Get into speech state.
        vad.update(&speech);
        vad.update(&speech);
        assert_eq!(vad.state(), VoiceActivity::Speech);

        // One silence chunk: not enough to transition.
        let (_, changed) = vad.update(&silence);
        assert!(!changed);
        assert_eq!(vad.state(), VoiceActivity::Speech);

        // Two silence chunks: still not enough.
        let (_, changed) = vad.update(&silence);
        assert!(!changed);

        // Third silence chunk: triggers transition back to silence.
        let (state, changed) = vad.update(&silence);
        assert_eq!(state, VoiceActivity::Silence);
        assert!(changed);
    }

    #[test]
    fn test_speech_interrupted_by_brief_silence() {
        let mut vad = SimpleVad::with_thresholds(0.01, 2, 5);
        let speech = make_speech(VAD_CHUNK_SIZE, 0.5);
        let silence = make_silence(VAD_CHUNK_SIZE);

        // Enter speech.
        vad.update(&speech);
        vad.update(&speech);
        assert_eq!(vad.state(), VoiceActivity::Speech);

        // Brief silence (only 2 chunks, need 5 to transition).
        vad.update(&silence);
        vad.update(&silence);
        assert_eq!(vad.state(), VoiceActivity::Speech);

        // Speech resumes — should stay in Speech.
        let (state, _) = vad.update(&speech);
        assert_eq!(state, VoiceActivity::Speech);
    }

    #[test]
    fn test_reset() {
        let mut vad = SimpleVad::with_thresholds(0.01, 1, 1);
        let speech = make_speech(VAD_CHUNK_SIZE, 0.5);

        vad.update(&speech);
        assert_eq!(vad.state(), VoiceActivity::Speech);

        vad.reset();
        assert_eq!(vad.state(), VoiceActivity::Silence);
    }

    #[test]
    fn test_compute_rms_empty() {
        assert_eq!(compute_rms(&[]), 0.0);
    }

    #[test]
    fn test_compute_rms_silence() {
        let silence = vec![0.0f32; 100];
        assert_eq!(compute_rms(&silence), 0.0);
    }

    #[test]
    fn test_compute_rms_signal() {
        let signal = vec![0.5f32; 100];
        let rms = compute_rms(&signal);
        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_process_chunk() {
        let signal = vec![0.3f32; 100];
        let rms = SimpleVad::process_chunk(&signal);
        assert!((rms - 0.3).abs() < 0.001);
    }
}
