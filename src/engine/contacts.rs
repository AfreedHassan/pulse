//! Contact name recognition.
//!
//! Helps with proper capitalization and formatting of contact names
//! in transcriptions. Names are stored in the local database.

use crate::storage::Storage;
use crate::types::{Contact, ContactCategory};

/// Contact name engine for improving name recognition in transcriptions.
pub struct ContactEngine<'a> {
    storage: &'a Storage,
    contacts: Vec<Contact>,
}

impl<'a> ContactEngine<'a> {
    pub fn new(storage: &'a Storage) -> Self {
        let contacts = storage.get_contacts().unwrap_or_default();
        Self { storage, contacts }
    }

    /// Refresh the contact list from storage.
    pub fn refresh(&mut self) {
        self.contacts = self.storage.get_contacts().unwrap_or_default();
    }

    /// Add a new contact.
    pub fn add_contact(&mut self, name: &str) {
        let contact = Contact::new(name.to_string(), ContactCategory::Acquaintance);
        if self.storage.save_contact(&contact).is_ok() {
            self.refresh();
        }
    }

    /// Apply proper capitalization for known contact names.
    ///
    /// Replaces case-insensitive matches with the properly-cased version.
    pub fn apply_names(&self, text: &str) -> String {
        let mut result = text.to_string();

        for contact in &self.contacts {
            let name = &contact.name;
            // Find case-insensitive occurrences and replace with proper casing.
            let lower_name = name.to_lowercase();
            let mut search_from = 0;

            while let Some(pos) = result[search_from..].to_lowercase().find(&lower_name) {
                let abs_pos = search_from + pos;
                let end_pos = abs_pos + lower_name.len();

                // Only replace if it's a word boundary.
                let is_word_start =
                    abs_pos == 0 || !result.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
                let is_word_end =
                    end_pos >= result.len() || !result.as_bytes()[end_pos].is_ascii_alphanumeric();

                if is_word_start && is_word_end {
                    result = format!("{}{}{}", &result[..abs_pos], name, &result[end_pos..]);
                }

                search_from = abs_pos + name.len();
                if search_from >= result.len() {
                    break;
                }
            }
        }

        result
    }

    /// Get the number of contacts.
    pub fn len(&self) -> usize {
        self.contacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_capitalization() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("John Smith");

        assert_eq!(
            engine.apply_names("I talked to john smith today"),
            "I talked to John Smith today"
        );
    }

    #[test]
    fn test_no_contacts() {
        let storage = Storage::in_memory().unwrap();
        let engine = ContactEngine::new(&storage);
        assert_eq!(engine.apply_names("hello world"), "hello world");
        assert!(engine.is_empty());
    }

    #[test]
    fn test_word_boundary() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Al");

        // Should NOT replace "al" inside "algorithm".
        assert_eq!(
            engine.apply_names("the algorithm works"),
            "the algorithm works"
        );

        // Should replace standalone "al".
        assert_eq!(engine.apply_names("talk to al"), "talk to Al");
    }

    #[test]
    fn test_add_and_count() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);

        engine.add_contact("Alice");
        engine.add_contact("Bob");

        assert_eq!(engine.len(), 2);
        assert!(!engine.is_empty());
    }

    #[test]
    fn test_case_insensitive_match() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Alice");

        assert_eq!(engine.apply_names("talk to ALICE"), "talk to Alice");
        assert_eq!(engine.apply_names("ALICE is here"), "Alice is here");
    }

    #[test]
    fn test_multiple_contacts() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Alice");
        engine.add_contact("Bob");

        assert_eq!(
            engine.apply_names("alice and bob went to the store"),
            "Alice and Bob went to the store"
        );
    }

    #[test]
    fn test_name_at_start() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Alice");

        assert_eq!(engine.apply_names("alice is here"), "Alice is here");
    }

    #[test]
    fn test_name_at_end() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Alice");

        assert_eq!(engine.apply_names("talk to alice"), "talk to Alice");
    }

    #[test]
    fn test_name_with_punctuation() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Alice");

        assert_eq!(engine.apply_names("Hello, alice!"), "Hello, Alice!");
        assert_eq!(engine.apply_names("alice?"), "Alice?");
    }

    #[test]
    fn test_multiple_occurrences() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Bob");

        assert_eq!(
            engine.apply_names("bob said that bob is coming"),
            "Bob said that Bob is coming"
        );
    }

    #[test]
    fn test_refresh_reloads_contacts() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);

        // Initially no contacts
        assert_eq!(engine.apply_names("talk to alice"), "talk to alice");

        // Add a contact directly to storage
        let contact = Contact::new("Alice".to_string(), ContactCategory::Close);
        storage.save_contact(&contact).unwrap();

        // Refresh should reload
        engine.refresh();
        assert_eq!(engine.apply_names("talk to alice"), "talk to Alice");
    }

    #[test]
    fn test_empty_text() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Alice");

        assert_eq!(engine.apply_names(""), "");
    }

    #[test]
    fn test_single_char_name() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Ed");

        assert_eq!(engine.apply_names("talk to ed"), "talk to Ed");
        assert_eq!(engine.apply_names("education"), "education");
    }

    #[test]
    fn test_name_with_numbers() {
        let storage = Storage::in_memory().unwrap();
        let mut engine = ContactEngine::new(&storage);
        engine.add_contact("Alice");

        // Numbers should be treated as word boundary
        assert_eq!(engine.apply_names("alice123"), "alice123");
        assert_eq!(engine.apply_names("123alice"), "123alice");
    }
}
