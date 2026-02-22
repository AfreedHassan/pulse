//! Contact types for context-aware writing.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContactCategory {
    Close,
    Professional,
    Acquaintance,
}

impl ContactCategory {
    pub fn all() -> &'static [ContactCategory] {
        &[
            ContactCategory::Close,
            ContactCategory::Professional,
            ContactCategory::Acquaintance,
        ]
    }

    pub fn suggested_tone(&self) -> &'static str {
        match self {
            ContactCategory::Close => "casual, friendly, informal",
            ContactCategory::Professional => "formal, respectful, concise",
            ContactCategory::Acquaintance => "polite, friendly, moderately formal",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    pub id: Uuid,
    pub name: String,
    pub category: ContactCategory,
    pub created_at: DateTime<Utc>,
}

impl Contact {
    pub fn new(name: String, category: ContactCategory) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            category,
            created_at: Utc::now(),
        }
    }

    pub fn close(name: String) -> Self {
        Self::new(name, ContactCategory::Close)
    }

    pub fn professional(name: String) -> Self {
        Self::new(name, ContactCategory::Professional)
    }

    pub fn acquaintance(name: String) -> Self {
        Self::new(name, ContactCategory::Acquaintance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_category_all() {
        let all = ContactCategory::all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_contact_category_suggested_tone() {
        assert!(ContactCategory::Close.suggested_tone().contains("casual"));
        assert!(ContactCategory::Professional
            .suggested_tone()
            .contains("formal"));
        assert!(ContactCategory::Acquaintance
            .suggested_tone()
            .contains("polite"));
    }

    #[test]
    fn test_contact_new() {
        let c = Contact::new("Alice".into(), ContactCategory::Close);
        assert_eq!(c.name, "Alice");
        assert_eq!(c.category, ContactCategory::Close);
        assert!(!c.id.is_nil());
    }

    #[test]
    fn test_contact_close() {
        let c = Contact::close("Bob".into());
        assert_eq!(c.category, ContactCategory::Close);
    }

    #[test]
    fn test_contact_professional() {
        let c = Contact::professional("Carol".into());
        assert_eq!(c.category, ContactCategory::Professional);
    }

    #[test]
    fn test_contact_acquaintance() {
        let c = Contact::acquaintance("Dave".into());
        assert_eq!(c.category, ContactCategory::Acquaintance);
    }

    #[test]
    fn test_contact_category_serde() {
        let cat = ContactCategory::Professional;
        let json = serde_json::to_string(&cat).unwrap();
        assert_eq!(json, "\"professional\"");
        let decoded: ContactCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, cat);
    }

    #[test]
    fn test_contact_serde() {
        let c = Contact::close("Alice".into());
        let json = serde_json::to_string(&c).unwrap();
        let decoded: Contact = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Alice");
        assert_eq!(decoded.category, ContactCategory::Close);
    }
}
