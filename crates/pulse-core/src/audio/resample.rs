//! Shared audio preprocessing: mono conversion, resampling, silence gating, trimming, normalization.

use tracing::debug;

/// Model's expected sample rate (16kHz).
pub const MODEL_SAMPLE_RATE: u32 = 16_000;

/// RMS below this threshold is treated as silence — skip inference entirely.
const SILENCE_RMS_THRESHOLD: f32 = 0.003;
/// Samples with absolute value below this are trimmed from leading/trailing edges.
const SILENCE_TRIM_THRESHOLD: f32 = 0.005;
/// Safety margin (in samples at 16kHz) to keep around trimmed speech boundaries.
/// 100ms = 1600 samples — preserves consonant onset/offset for short utterances.
const TRIM_PAD_SAMPLES: usize = 1600;
/// Peak amplitude target for normalization.
const NORMALIZE_TARGET: f32 = 0.9;

/// Full audio preprocessing pipeline: mono conversion, resample to 16kHz,
/// silence gate, trim, and normalize. Returns `None` if the audio is silence.
pub fn preprocess_audio(pcm: &[f32], sample_rate: u32, channels: u16) -> Option<Vec<f32>> {
    // 1. Convert to mono.
    let mono: Vec<f32> = if channels == 1 {
        pcm.to_vec()
    } else {
        let ch = channels as usize;
        pcm.chunks(ch)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // 2. Resample to 16kHz if needed.
    let pcm_16k = if sample_rate == MODEL_SAMPLE_RATE {
        mono
    } else {
        resample_sinc(&mono, sample_rate, MODEL_SAMPLE_RATE)
    };

    if pcm_16k.is_empty() {
        return None;
    }

    // 3. Silence gate.
    let rms = compute_rms(&pcm_16k);
    debug!("[audio] {} samples, RMS={:.6}", pcm_16k.len(), rms);
    if rms < SILENCE_RMS_THRESHOLD {
        debug!("[audio] below silence threshold, skipping");
        return None;
    }

    // 4. Trim leading/trailing silence (with safety margin to preserve speech edges).
    let first_voice = pcm_16k
        .iter()
        .position(|s| s.abs() > SILENCE_TRIM_THRESHOLD)
        .unwrap_or(pcm_16k.len());
    let last_voice = pcm_16k
        .iter()
        .rposition(|s| s.abs() > SILENCE_TRIM_THRESHOLD)
        .map_or(first_voice, |i| i + 1);

    if first_voice >= last_voice {
        return None;
    }

    let start = first_voice.saturating_sub(TRIM_PAD_SAMPLES);
    let end = (last_voice + TRIM_PAD_SAMPLES).min(pcm_16k.len());

    let mut trimmed = pcm_16k[start..end].to_vec();
    debug!("[audio] after trim: {} samples", trimmed.len());

    // 5. Peak normalize.
    let peak = trimmed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let gain = NORMALIZE_TARGET / peak;
        trimmed.iter_mut().for_each(|s| *s *= gain);
    }

    Some(trimmed)
}

/// Windowed-sinc (Lanczos) resampling with anti-aliasing.
///
/// Uses a sinc kernel with a Lanczos window to properly low-pass filter
/// the signal before decimation, preventing aliasing artifacts. The kernel
/// half-width of 8 samples provides good quality for speech audio.
pub fn resample_sinc(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    // Anti-aliasing: when downsampling, scale the cutoff frequency down.
    let cutoff = if to_rate < from_rate {
        ratio // e.g. 16000/48000 = 0.333
    } else {
        1.0
    };

    // Kernel half-width in source samples. 8 lobes is good for speech.
    let half_width: usize = 8;

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let center = src_pos.floor() as i64;
        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for j in (center - half_width as i64 + 1)..=(center + half_width as i64) {
            if j < 0 || j >= samples.len() as i64 {
                continue;
            }
            let x = (src_pos - j as f64) * cutoff;
            let w = lanczos_kernel(x, half_width as f64);
            sum += samples[j as usize] as f64 * w * cutoff;
            weight_sum += w * cutoff;
        }

        let sample = if weight_sum > 0.0 {
            (sum / weight_sum) as f32
        } else {
            0.0
        };
        output.push(sample);
    }

    output
}

/// Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, else 0.
fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else if x.abs() >= a {
        0.0
    } else {
        let px = std::f64::consts::PI * x;
        let pxa = std::f64::consts::PI * x / a;
        (px.sin() / px) * (pxa.sin() / pxa)
    }
}

/// Compute RMS energy of audio samples.
pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_sinc_same_rate() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let resampled = resample_sinc(&samples, 44100, 44100);
        assert_eq!(resampled, samples);
    }

    #[test]
    fn test_resample_sinc_downsample() {
        let samples: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.01).sin()).collect();
        let resampled = resample_sinc(&samples, 44100, 16000);
        assert!((resampled.len() as i64 - 16000).abs() < 2);
    }

    #[test]
    fn test_resample_sinc_empty() {
        let resampled = resample_sinc(&[], 44100, 16000);
        assert!(resampled.is_empty());
    }

    #[test]
    fn test_compute_rms_silence() {
        assert_eq!(compute_rms(&vec![0.0; 100]), 0.0);
    }

    #[test]
    fn test_compute_rms_signal() {
        let rms = compute_rms(&vec![0.5; 100]);
        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_preprocess_audio_silence() {
        let silence = vec![0.0f32; 16000];
        assert!(preprocess_audio(&silence, 16000, 1).is_none());
    }

    #[test]
    fn test_preprocess_audio_mono_conversion() {
        // Stereo signal: L=0.5, R=0.5 -> mono=0.5
        let stereo: Vec<f32> = (0..32000)
            .map(|i| {
                let t = (i / 2) as f32 / 16000.0;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();
        let result = preprocess_audio(&stereo, 16000, 2);
        assert!(result.is_some());
    }
}
