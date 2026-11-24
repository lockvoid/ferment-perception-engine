//! Each `ComputeUnits` a placement names is the `MLComputeUnits` Core ML holds
//! for the loaded model. `coreml-native` 0.2.0 sent `CpuOnly` as `cpuAndGPU`
//! and `CpuAndNeuralEngine` as `all`, so every CPU pin in the fleet ran with
//! the GPU allowed and every Neural Engine pin with the GPU too; the vendored
//! crate (`vendor/coreml-native`) is the fix, and this is what holds it.

#![cfg(all(feature = "coreml", target_vendor = "apple"))]

use ferment_engine_infer::ComputeUnits;

const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../vendor/coreml-native/tests/fixtures/test_linear.mlmodelc"
);

/// Apple's `MLComputeUnits` raw values.
const SELECTIONS: [(ComputeUnits, isize); 4] = [
    (ComputeUnits::CpuOnly, 0),
    (ComputeUnits::CpuAndGpu, 1),
    (ComputeUnits::All, 2),
    (ComputeUnits::CpuAndNeuralEngine, 3),
];

#[test]
fn a_loaded_model_holds_the_units_it_was_asked_for() {
    for (units, raw) in SELECTIONS {
        let model = coreml_native::Model::load(MODEL, units).unwrap();
        assert_eq!(model.ml_compute_units(), raw, "{units}");
    }
}

#[test]
fn an_asynchronously_loaded_model_holds_the_units_it_was_asked_for() {
    for (units, raw) in SELECTIONS {
        let model = coreml_native::Model::load_async(MODEL, units)
            .unwrap()
            .block_on()
            .unwrap();
        assert_eq!(model.ml_compute_units(), raw, "{units}");
    }
}
