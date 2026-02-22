//! Writing mode types for transcription style control.

use serde::{Deserialize, Serialize};

use super::app::AppCategory;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WritingMode {
    Formal,
    #[default]
    Casual,
    VeryCasual,
    Excited,
}

impl WritingMode {
    pub fn prompt_modifier(&self) -> &'static str {
        match self {
            Self::Formal => {
                "Reformat in formal, professional tone. Replace casual phrases with proper \
                 equivalents: \"gonna\" → \"going to\", \"wanna\" → \"want to\", \
                 \"5 min\" → \"five minutes\", \"info\" → \"information\", \"asap\" → \
                 \"as soon as possible\". Transform slang into professional alternatives. \
                 Use complete sentences, proper grammar, and polished language. \
                 Output EXACTLY as it would be typed—nothing more, nothing else."
            }
            Self::Casual => {
                "Reformat in friendly, conversational tone. Keep contractions (\"I'm\", \
                 \"don't\", \"we'll\"), use natural language, but ensure it's clear and warm. \
                 Numbers can stay as digits (\"5 minutes\" not \"five minutes\"). \
                 Preserve the intended meaning exactly. \
                 Output EXACTLY as it would be typed—do NOT add commentary or anything beyond \
                 the reformatted text."
            }
            Self::VeryCasual => {
                "Reformat in casual texting style. Use all lowercase except for proper nouns \
                 (names, places, brands). Minimal punctuation — skip periods at the end of \
                 messages, skip commas unless needed for clarity. Use abbreviations: \
                 \"going to\" → \"gonna\", \"right now\" → \"rn\", \"sorry\" → \"sry\", \
                 \"because\" → \"bc\", \"about\" → \"abt\". Keep it brief and informal like \
                 a text to a close friend. \
                 Output EXACTLY as it would be typed—nothing else."
            }
            Self::Excited => {
                "Reformat with enthusiasm and warmth. Add exclamation marks where appropriate, \
                 express energy and affection. Transform neutral phrasing into excited \
                 equivalents: \"that's good\" → \"that's amazing!\", \"thanks\" → \"thank \
                 you so much!\". Make it sound excited while preserving the intended meaning. \
                 Output EXACTLY as it would be typed—nothing more."
            }
        }
    }

    pub fn all() -> &'static [WritingMode] {
        &[
            WritingMode::Formal,
            WritingMode::Casual,
            WritingMode::VeryCasual,
            WritingMode::Excited,
        ]
    }

    pub fn suggested_for_category(category: AppCategory) -> Self {
        match category {
            AppCategory::Email | AppCategory::Code | AppCategory::Documents => WritingMode::Formal,
            AppCategory::Slack | AppCategory::Browser => WritingMode::Casual,
            AppCategory::Imessage | AppCategory::Social | AppCategory::Terminal => {
                WritingMode::VeryCasual
            }
            AppCategory::Unknown => WritingMode::Casual,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "formal" => Some(Self::Formal),
            "casual" => Some(Self::Casual),
            "very_casual" | "verycasual" => Some(Self::VeryCasual),
            "excited" => Some(Self::Excited),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writing_mode_default() {
        assert_eq!(WritingMode::default(), WritingMode::Casual);
    }

    #[test]
    fn test_writing_mode_parse() {
        assert_eq!(WritingMode::parse("formal"), Some(WritingMode::Formal));
        assert_eq!(WritingMode::parse("CASUAL"), Some(WritingMode::Casual));
        assert_eq!(
            WritingMode::parse("very-casual"),
            Some(WritingMode::VeryCasual)
        );
        assert_eq!(
            WritingMode::parse("very_casual"),
            Some(WritingMode::VeryCasual)
        );
        assert_eq!(WritingMode::parse("excited"), Some(WritingMode::Excited));
        assert_eq!(WritingMode::parse("unknown"), None);
    }

    #[test]
    fn test_writing_mode_prompt_modifier() {
        for mode in WritingMode::all() {
            let modifier = mode.prompt_modifier();
            assert!(!modifier.is_empty());
            assert!(modifier.contains("EXACTLY"));
        }
    }

    #[test]
    fn test_writing_mode_suggested_for_category() {
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Email),
            WritingMode::Formal
        );
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Code),
            WritingMode::Formal
        );
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Documents),
            WritingMode::Formal
        );
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Slack),
            WritingMode::Casual
        );
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Imessage),
            WritingMode::VeryCasual
        );
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Browser),
            WritingMode::Casual
        );
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Social),
            WritingMode::VeryCasual
        );
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Terminal),
            WritingMode::VeryCasual
        );
        assert_eq!(
            WritingMode::suggested_for_category(AppCategory::Unknown),
            WritingMode::Casual
        );
    }

    #[test]
    fn test_writing_mode_all() {
        let all = WritingMode::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&WritingMode::Formal));
        assert!(all.contains(&WritingMode::Casual));
        assert!(all.contains(&WritingMode::VeryCasual));
        assert!(all.contains(&WritingMode::Excited));
    }

    #[test]
    fn test_writing_mode_serde() {
        let mode = WritingMode::VeryCasual;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"very_casual\"");
        let decoded: WritingMode = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, mode);
    }
}
