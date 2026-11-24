use ferment_engine_retrieval::{EmbeddingSpace, GlapFrontend, siglip2_pixels};
use ferment_engine_test_fixtures::export;

#[test]
fn spaces_match_the_exported_contracts() {
    for (family, space) in [
        ("glap", EmbeddingSpace::GlapWindowV1),
        ("siglip2", EmbeddingSpace::Siglip2Base224V1),
    ] {
        let Some(path) = export::path(&format!("{family}/model.json")) else {
            return;
        };
        let contract: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(contract["space"]["id"], space.id());
        assert_eq!(contract["space"]["dimensions"], space.dimensions());
        assert_eq!(contract["space"]["normalization"], "l2");
        assert_eq!(contract["space"]["metric"], "cosine");
    }
}

#[test]
fn glap_mel_is_the_stock_frontend_without_attention_padding() {
    if export::path("glap/goldens/cases.json").is_none() {
        return;
    }
    let frontend = GlapFrontend::new().unwrap();
    for name in ["music", "speech", "one-shot", "silence"] {
        let samples = export::require_f32_le(&format!("glap/goldens/{name}.audio.f32"));
        let golden = export::require_f32_le(&format!("glap/goldens/{name}.mel.f32"));
        let actual = frontend.prepare(&samples).unwrap().to_f32().unwrap();
        assert_eq!(actual.len(), golden.len());
        let peak = golden.iter().copied().fold(f32::MIN, f32::max);
        let (mut loud, mut quiet) = (0.0_f32, 0.0_f32);
        for (&a, &b) in actual.iter().zip(&golden) {
            if b >= peak - 60.0 {
                loud = loud.max((a - b).abs());
            } else {
                quiet = quiet.max((a - b).abs());
            }
        }
        // Same Torch-vs-realfft bounds as the shared Dasheng frontend. Quiet
        // log-floor bins amplify FFT roundoff; end-to-end embeddings are also
        // graded independently by the production-model oracle.
        eprintln!("[glap-mel] {name}: loud={loud:e}, quiet={quiet:e}");
        assert!(loud < 1e-3, "{name}: {loud}");
        assert!(quiet < 5e-2, "{name}: {quiet}");
    }
}

#[test]
fn siglip2_resize_and_normalization_match_stock_pixels() {
    if export::path("siglip2/goldens/cases.json").is_none() {
        return;
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&export::require_bytes("siglip2/goldens/cases.json")).unwrap();
    for case in manifest["image"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let width = case["width"].as_u64().unwrap() as u32;
        let height = case["height"].as_u64().unwrap() as u32;
        let rgb = export::require_bytes(&format!("siglip2/goldens/{name}.rgb.u8"));
        let golden = export::require_f32_le(&format!("siglip2/goldens/{name}.pixels.f32"));
        let actual = siglip2_pixels(&rgb, width, height)
            .unwrap()
            .to_f32()
            .unwrap();
        assert_eq!(actual.len(), golden.len());
        let delta = actual
            .iter()
            .zip(&golden)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("[siglip2-pixels] {name}: max_abs={delta:e}");
        assert!(delta < 1e-6, "{name}: {delta}");
    }
}

#[test]
fn invalid_rgb_is_rejected() {
    for (bytes, w, h) in [
        (&[][..], 0, 0),
        (&[0][..], 1, 1),
        (&[0, 0, 0][..], u32::MAX, u32::MAX),
    ] {
        assert!(siglip2_pixels(bytes, w, h).is_err());
    }
}
