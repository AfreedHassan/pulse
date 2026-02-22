//! Completion provider trait and types.

use crate::error::Result;
use async_trait::async_trait;

/// Request for text completion/formatting.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// System prompt.
    pub system: String,
    /// User message (the text to format).
    pub user: String,
    /// Temperature (0.0 = deterministic).
    pub temperature: f32,
}

/// Response from a completion provider.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The completed/formatted text.
    pub text: String,
}

/// Trait for LLM completion providers.
#[async_trait]
pub trait CompletionProvider: Send + Sync {
    /// Provider name for logging/display.
    fn name(&self) -> &'static str;

    /// Complete/format text.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    /// Whether this provider is properly configured.
    fn is_configured(&self) -> bool;
}
