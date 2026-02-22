use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::time::Duration;

use crate::types::WritingMode;

const TIMEOUT: Duration = Duration::from_secs(10);

const SYSTEM_PROMPT: &str = "\
You are a text formatting function inside Pulse, a local voice dictation app (similar to Wispr Flow). \
You receive raw speech-to-text transcripts and return cleaned-up versions. \
The word \"pulse\" in the transcript almost always refers to the app name \"Pulse\" — capitalize it accordingly.

Rules:
- Fix capitalization, punctuation, and spacing
- If the speaker is listing items or steps, use a bulleted or numbered list
- Otherwise keep it as prose
- If the speaker corrects themselves (e.g. \"no wait\", \"scratch that\", \"actually\", \"I mean\"), apply the correction and omit the original mistake and the correction phrase
- Preserve the speaker's intended words — do not add or rephrase beyond corrections
- Output ONLY the formatted text

Example input:
i went to the store and it was pretty busy but i managed to find everything i needed

Example output:
I went to the store and it was pretty busy, but I managed to find everything I needed.

Example input:
so i need to get milk eggs bread and also pick up the dry cleaning

Example output:
So I need to get:
- Milk
- Eggs
- Bread

And also pick up the dry cleaning.

Example input:
hey john can we meet at 6pm tomorrow uh no scratch that lets meet at 10pm tomorrow

Example output:
Hey John, can we meet at 10pm tomorrow?

Example input:
okay there are three things we need to do first we need to fix the login bug second we need to update the docs and third we need to deploy to staging

Example output:
Okay, there are three things we need to do:
1. First, we need to fix the login bug.
2. Second, we need to update the docs.
3. Third, we need to deploy to staging.";

/// Preambles that indicate the model is responding conversationally instead of formatting.
const BAD_PREFIXES: &[&str] = &[
    "Sure",
    "Here",
    "Of course",
    "I'd",
    "I'll",
    "Let me",
    "The formatted",
    "Below",
    "Certainly",
];

/// Max ratio of output length to input length before we consider it hallucination.
const MAX_LENGTH_RATIO: f64 = 2.0;

pub struct FormatterConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl FormatterConfig {
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("LLM_BASE_URL").ok()?;
        let api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        Some(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
        })
    }
}

pub struct Formatter {
    client: reqwest::blocking::Client,
    config: FormatterConfig,
}

impl Formatter {
    pub fn new(config: FormatterConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(TIMEOUT)
            .build()
            .expect("Failed to build HTTP client");
        Self { client, config }
    }

    /// Stream formatted text directly to stdout. Falls back to printing raw text on error.
    pub fn format_to_stdout(&self, text: &str, context: Option<&str>) {
        match self.format(text, context) {
            Ok(formatted) => {
                if passes_guardrails(text, &formatted) {
                    println!("{}", formatted);
                } else {
                    eprintln!("Formatter produced bad output, using raw text");
                    println!("{}", text);
                }
            }
            Err(e) => {
                eprintln!("Formatter error (using raw text): {}", e);
                println!("{}", text);
            }
        }
    }

    /// Format text using the default style.
    pub fn format(&self, text: &str, context: Option<&str>) -> Result<String> {
        self.format_with_mode(text, None, context)
    }

    /// Format text with an optional writing mode.
    pub fn format_with_mode(
        &self,
        text: &str,
        mode: Option<&WritingMode>,
        context: Option<&str>,
    ) -> Result<String> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let mut system = SYSTEM_PROMPT.to_string();
        if let Some(m) = mode {
            system.push_str(&format!("\n\nAdditional style: {}", m.prompt_modifier()));
        }

        // Append text field context if available.
        if let Some(ctx) = context {
            if !ctx.is_empty() {
                system.push_str(&format!(
                    "\n\nThe user is typing in a text field that already contains:\n---\n{}\n---\nFormat the new dictated text so it flows naturally with the existing content. Consider capitalization, punctuation, and continuity.",
                    ctx
                ));
            }
        }

        let messages = vec![
            Message {
                role: "system",
                content: system,
            },
            Message {
                role: "user",
                content: text.to_string(),
            },
        ];

        if let Ok(result) = self.try_stream(&url, &messages) {
            if !result.is_empty() {
                return Ok(result);
            }
        }

        self.try_non_stream(&url, &messages)
    }

    /// Build a POST request with optional bearer auth.
    fn post(&self, url: &str) -> reqwest::blocking::RequestBuilder {
        let req = self.client.post(url);
        if self.config.api_key.is_empty() {
            req
        } else {
            req.bearer_auth(&self.config.api_key)
        }
    }

    fn try_stream(&self, url: &str, messages: &[Message]) -> Result<String> {
        let body = ChatRequest {
            model: &self.config.model,
            messages,
            temperature: 0.0,
            stream: true,
        };

        let resp = self.post(url).json(&body).send()?.error_for_status()?;
        let reader = BufReader::new(resp);
        let mut result = String::new();

        for line in reader.lines() {
            let line = line?;
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                if let Some(delta) = chunk.choices.into_iter().next() {
                    if let Some(content) = delta.delta.content {
                        result.push_str(&content);
                    }
                }
            }
        }

        if result.is_empty() {
            bail!("Empty stream response");
        }

        Ok(result.trim().to_string())
    }

    fn try_non_stream(&self, url: &str, messages: &[Message]) -> Result<String> {
        let body = ChatRequest {
            model: &self.config.model,
            messages,
            temperature: 0.0,
            stream: false,
        };

        let resp = self.post(url).json(&body).send()?.error_for_status()?;
        let raw = resp.text()?;
        eprintln!(
            "[formatter debug] raw response: {}",
            &raw[..raw.len().min(500)]
        );

        let chat_resp: NonStreamResponse = serde_json::from_str(&raw)?;

        let content = chat_resp
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        if content.is_empty() {
            bail!("Empty response from LLM");
        }

        Ok(content.trim().to_string())
    }
}

/// Minimum fraction of input words that must appear in the output.
const MIN_WORD_OVERLAP: f64 = 0.5;

/// Check that the LLM output looks like formatted text, not a chatbot response.
pub fn passes_guardrails(input: &str, output: &str) -> bool {
    // Reject if it starts with a conversational preamble.
    if BAD_PREFIXES.iter().any(|prefix| output.starts_with(prefix)) {
        return false;
    }

    // Reject if output is way longer than input (model is hallucinating/explaining).
    let ratio = output.len() as f64 / input.len().max(1) as f64;
    if ratio > MAX_LENGTH_RATIO {
        return false;
    }

    // Reject if the model added more than one extra question mark (allows punctuating
    // an existing question, but catches the model asking follow-up questions).
    let input_questions = input.chars().filter(|&c| c == '?').count();
    let output_questions = output.chars().filter(|&c| c == '?').count();
    if output_questions > input_questions + 1 {
        return false;
    }

    // Reject if most input words don't appear in the output (model rephrased or hallucinated).
    let input_words: Vec<&str> = input
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();
    if !input_words.is_empty() {
        let output_lower = output.to_lowercase();
        let matches = input_words
            .iter()
            .filter(|w| output_lower.contains(&w.to_lowercase()))
            .count();
        let overlap = matches as f64 / input_words.len() as f64;
        if overlap < MIN_WORD_OVERLAP {
            return false;
        }
    }

    true
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    temperature: f32,
    stream: bool,
}

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: Delta,
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct NonStreamResponse {
    choices: Vec<NonStreamChoice>,
}

#[derive(Deserialize)]
struct NonStreamChoice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passes_guardrails_basic() {
        let input = "hello world";
        let output = "Hello, world.";
        assert!(passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_preserves_words() {
        let input = "the quick brown fox jumps over the lazy dog";
        let output = "The quick brown fox jumps over the lazy dog.";
        assert!(passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_sure_prefix() {
        let input = "hello";
        let output = "Sure! Hello.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_here_prefix() {
        let input = "test";
        let output = "Here is the formatted text: Test.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_of_course_prefix() {
        let input = "hi";
        let output = "Of course! Hi.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_id_prefix() {
        let input = "hello";
        let output = "I'd be happy to help! Hello.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_ill_prefix() {
        let input = "test";
        let output = "I'll format that for you. Test.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_let_me_prefix() {
        let input = "hello";
        let output = "Let me help you with that. Hello.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_formatted_prefix() {
        let input = "test";
        let output = "The formatted text is: Test.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_below_prefix() {
        let input = "hello";
        let output = "Below is the result: Hello.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_certainly_prefix() {
        let input = "test";
        let output = "Certainly! Test.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_long_output() {
        let input = "hi";
        let output = "Hi there! I hope you're having a wonderful day. Is there anything I can help you with today? Feel free to ask any questions you might have.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_allows_one_extra_question() {
        let input = "can you help me";
        let output = "Can you help me?";
        assert!(passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_multiple_extra_questions() {
        let input = "i need to fix the bug";
        let output = "I need to fix the bug. Would you like help? What kind of bug is it?";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_rejects_low_word_overlap() {
        let input = "pick up groceries after work";
        let output = "Remember to complete your shopping tasks this evening.";
        assert!(!passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_accepts_list_formatting() {
        let input = "first fix the bug second update docs third deploy";
        let output = "1. First, fix the bug.\n2. Second, update docs.\n3. Third, deploy.";
        assert!(passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_empty_input() {
        let input = "";
        let output = "";
        assert!(passes_guardrails(input, output));
    }

    #[test]
    fn test_passes_guardrails_same_input_output() {
        let input = "hello world";
        let output = "hello world";
        assert!(passes_guardrails(input, output));
    }

    #[test]
    fn test_formatter_config_defaults() {
        unsafe {
            std::env::set_var("LLM_BASE_URL", "http://localhost:11434");
        }
        let config = FormatterConfig::from_env().unwrap();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "gpt-4o-mini");
        unsafe {
            std::env::remove_var("LLM_BASE_URL");
        }
    }

    #[test]
    fn test_formatter_config_with_model() {
        unsafe {
            std::env::set_var("LLM_BASE_URL", "http://localhost:11434");
            std::env::set_var("LLM_MODEL", "llama3");
        }
        let config = FormatterConfig::from_env().unwrap();
        assert_eq!(config.model, "llama3");
        unsafe {
            std::env::remove_var("LLM_BASE_URL");
            std::env::remove_var("LLM_MODEL");
        }
    }

    #[test]
    fn test_formatter_config_missing_url() {
        unsafe {
            std::env::remove_var("LLM_BASE_URL");
        }
        assert!(FormatterConfig::from_env().is_none());
    }

    #[test]
    fn test_formatter_config_trims_trailing_slash() {
        unsafe {
            std::env::set_var("LLM_BASE_URL", "http://localhost:11434/");
        }
        let config = FormatterConfig::from_env().unwrap();
        assert_eq!(config.base_url, "http://localhost:11434");
        unsafe {
            std::env::remove_var("LLM_BASE_URL");
        }
    }
}
