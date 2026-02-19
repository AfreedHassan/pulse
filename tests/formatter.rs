use pulse::engine::formatter::{passes_guardrails, Formatter, FormatterConfig};

fn make_formatter() -> Formatter {
    dotenvy::dotenv().ok();
    let config =
        FormatterConfig::from_env().expect("LLM_BASE_URL must be set to run formatter tests");
    Formatter::new(config)
}

// ── Guardrail unit tests (no LLM needed) ────────────────────────────

#[test]
fn test_guardrails_accepts_good_formatting() {
    let input = "so i need eggs milk and bread";
    let output = "So I need eggs, milk, and bread.";
    assert!(passes_guardrails(input, output));
}

#[test]
fn test_guardrails_rejects_preamble() {
    let input = "hello world";
    let output = "Sure! Here is the formatted text: Hello, world.";
    assert!(!passes_guardrails(input, output));
}

#[test]
fn test_guardrails_rejects_hallucination() {
    let input = "buy eggs";
    let output = "I'd be happy to help you with your grocery list! Here are some suggestions for what to buy along with eggs: milk, bread, butter, and cheese.";
    assert!(!passes_guardrails(input, output));
}

#[test]
fn test_guardrails_allows_punctuating_a_question() {
    let input = "can you help me with the bug";
    let output = "Can you help me with the bug?";
    assert!(passes_guardrails(input, output));
}

#[test]
fn test_guardrails_rejects_added_followup_questions() {
    let input = "i need to fix the bug";
    let output = "I need to fix the bug. Would you like me to help? What kind of bug is it?";
    assert!(!passes_guardrails(input, output));
}

#[test]
fn test_guardrails_rejects_low_word_overlap() {
    let input = "pick up groceries after work";
    let output = "Remember to complete your shopping tasks this evening.";
    assert!(!passes_guardrails(input, output));
}

#[test]
fn test_guardrails_accepts_list_formatting() {
    let input = "first fix the bug second update docs third deploy";
    let output = "1. First, fix the bug.\n2. Second, update docs.\n3. Third, deploy.";
    assert!(passes_guardrails(input, output));
}

#[test]
fn test_guardrails_rejects_very_long_output() {
    let input = "hello";
    let output = "Hello there! I hope you're having a wonderful day. Is there anything I can help you with today? Feel free to ask any questions.";
    assert!(!passes_guardrails(input, output));
}

#[test]
fn test_guardrails_preserves_numbers() {
    let input = "call me at 555 123 4567";
    let output = "Call me at 555-123-4567.";
    assert!(passes_guardrails(input, output));
}

// ── LLM integration tests (require LLM_BASE_URL) ───────────────────

#[test]
#[ignore] // requires LLM_BASE_URL and LLM_API_KEY
fn test_formatter_prose() {
    let f = make_formatter();
    let input = "so i was thinking we could maybe go to the store tomorrow and pick up some stuff for the party";
    let result = f.format(input).expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);

    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
    // Should have a capital S at start
    assert!(
        result.starts_with("So") || result.starts_with("so"),
        "Expected output to start with the original first word"
    );
    // Should end with punctuation
    let last = result.trim_end().chars().last().unwrap();
    assert!(
        last == '.' || last == '!' || last == '?',
        "Expected trailing punctuation, got '{}'",
        last
    );
}

#[test]
#[ignore]
fn test_formatter_grocery_list() {
    let f = make_formatter();
    let input = "So I need to get a dozen eggs a dozen bananas some apples and a kilogram of flour tomorrow. That's my grocery list.";
    let result = f.format(input).expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);

    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
    // Key words must be preserved
    for word in ["eggs", "bananas", "apples", "flour"] {
        assert!(
            result.to_lowercase().contains(word),
            "Missing word '{}' in output: {}",
            word,
            result
        );
    }
    // Should be formatted as a list
    let has_list = result.contains("- ")
        || result.contains("• ")
        || result.contains("1.")
        || result.contains("1)");
    assert!(has_list, "Expected list formatting in: {}", result);
}

#[test]
#[ignore]
fn test_formatter_numbered_steps() {
    let f = make_formatter();
    let input = "okay there are three things we need to do first we need to fix the login bug second we need to update the docs and third we need to deploy to staging";
    let result = f.format(input).expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);

    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
    // Should contain numbered items or bullet points
    let has_list = result.contains("1.")
        || result.contains("1)")
        || result.contains("- ")
        || result.contains("• ");
    assert!(has_list, "Expected list formatting in: {}", result);
}

#[test]
#[ignore]
fn test_formatter_does_not_answer_questions() {
    let f = make_formatter();
    let input = "can you help me figure out what to do about the database migration";
    let result = f.format(input).expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);

    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
    // Must not start with chatbot preambles
    let lower = result.to_lowercase();
    assert!(
        !lower.starts_with("sure"),
        "Model answered instead of formatting: {}",
        result
    );
    assert!(
        !lower.starts_with("i'd"),
        "Model answered instead of formatting: {}",
        result
    );
    assert!(
        !lower.starts_with("of course"),
        "Model answered instead of formatting: {}",
        result
    );
}

#[test]
#[ignore]
fn test_formatter_self_correction() {
    let f = make_formatter();
    let input =
        "hey john can we meet at 6pm tomorrow uh no scratch that lets meet at 10pm tomorrow";
    let result = f.format(input).expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);

    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
    // Should contain the corrected time, not the original
    assert!(
        result.contains("10"),
        "Expected corrected time '10pm' in: {}",
        result
    );
    assert!(
        !result.contains("6pm") && !result.contains("6 pm"),
        "Should have dropped the original '6pm' in: {}",
        result
    );
    // Correction phrases should be gone
    let lower = result.to_lowercase();
    assert!(
        !lower.contains("scratch that"),
        "Correction phrase still present: {}",
        result
    );
    assert!(!lower.contains("uh no"), "Filler still present: {}", result);
}

#[test]
#[ignore]
fn test_formatter_short_input() {
    let f = make_formatter();
    let input = "hello";
    let result = f.format(input).expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);

    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
    assert!(
        result.to_lowercase().contains("hello"),
        "Lost the word 'hello' in: {}",
        result
    );
}

#[test]
#[ignore]
fn test_formatter_with_mode_formal() {
    use pulse::types::WritingMode;
    let f = make_formatter();
    let input = "gonna grab some food";
    let result = f
        .format_with_mode(input, Some(&WritingMode::Formal))
        .expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);
    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
    assert!(
        result.to_lowercase().contains("going to"),
        "Expected formal 'going to' in: {}",
        result
    );
}

#[test]
#[ignore]
fn test_formatter_with_mode_casual() {
    use pulse::types::WritingMode;
    let f = make_formatter();
    let input = "i am going to the store";
    let result = f
        .format_with_mode(input, Some(&WritingMode::Casual))
        .expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);
    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
}

#[test]
#[ignore]
fn test_formatter_with_mode_very_casual() {
    use pulse::types::WritingMode;
    let f = make_formatter();
    let input = "sorry about that";
    let result = f
        .format_with_mode(input, Some(&WritingMode::VeryCasual))
        .expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);
    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
}

#[test]
#[ignore]
fn test_formatter_with_mode_excited() {
    use pulse::types::WritingMode;
    let f = make_formatter();
    let input = "that is so cool";
    let result = f
        .format_with_mode(input, Some(&WritingMode::Excited))
        .expect("format failed");
    eprintln!("Input:  {}", input);
    eprintln!("Output: {}", result);
    assert!(
        passes_guardrails(input, &result),
        "Guardrails failed on: {}",
        result
    );
    assert!(
        result.contains("!"),
        "Expected exclamation in excited mode: {}",
        result
    );
}
