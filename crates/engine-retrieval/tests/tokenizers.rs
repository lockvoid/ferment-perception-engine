use ferment_engine_retrieval::{GlapLanguage, GlapTokenizer, Siglip2Tokenizer};
use ferment_engine_test_fixtures::export;

#[test]
#[ignore = "requires the two hash-verified tokenizer artifacts from ferment-models-export"]
fn token_ids_match_pinned_python_tokenizers_including_unicode_and_truncation() {
    let glap = GlapTokenizer::from_path(
        &export::artifact("glap_sentencepiece.model").expect("missing GLAP tokenizer"),
    )
    .unwrap();
    let siglip = Siglip2Tokenizer::from_path(
        &export::artifact("siglip2_tokenizer.json").expect("missing SigLIP tokenizer"),
    )
    .unwrap();
    for family in ["glap", "siglip2"] {
        let manifest: serde_json::Value = serde_json::from_slice(&export::require_bytes(&format!(
            "{family}/goldens/cases.json"
        )))
        .unwrap();
        for case in manifest["tokenization"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let expected: Vec<i32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let actual = if family == "glap" {
                let lang: GlapLanguage = serde_json::from_value(case["language"].clone()).unwrap();
                glap.encode(text, lang).unwrap()
            } else {
                siglip.encode(text).unwrap()
            };
            assert_eq!(actual, expected, "{family}: {text}");
        }
    }
    assert!(serde_json::from_str::<GlapLanguage>("\"unknown\"").is_err());
    assert!(siglip.encode(&"x".repeat(1_048_577)).is_err());
    assert!(
        glap.encode(&"x".repeat(1_048_577), GlapLanguage::En)
            .is_err()
    );
}
