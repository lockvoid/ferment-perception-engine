//! Small real-ORT controls for the production Parakeet session policy.
//! The source graph deliberately puts every initializer Cast before its users,
//! as the admitted Parakeet export does. These are not model-quality goldens.

use super::{ModelId, OrtBackend, OrtExecutionProvider};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferment-parakeet-policy-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while value >= 128 {
        out.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    out.push(value as u8);
    out
}
fn integer(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}
fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(value.len() as u64),
        value.to_vec(),
    ]
    .concat()
}
fn text(field: u64, value: &str) -> Vec<u8> {
    bytes(field, value.as_bytes())
}
fn value_info(name: &str) -> Vec<u8> {
    let shape = [bytes(1, &integer(1, 1)), bytes(1, &integer(1, 4))].concat();
    let tensor_type = [integer(1, 1), bytes(2, &shape)].concat();
    [text(1, name), bytes(2, &bytes(1, &tensor_type))].concat()
}

/// ONNX IR8/opset17, FLOAT input/output [1,4]. Stored IEEE half bit patterns:
/// a=[.5,-2,4,.25], b=[-2,.25,.5,4], c=[.125,4,-2,-.5].
/// x=[8,-4,2,-8] therefore gives exactly [-1,8,-8,4] through three Mul nodes.
fn model_bytes() -> Vec<u8> {
    let names = ["a", "b", "c"];
    let weights: [[u16; 4]; 3] = [
        [0x3800, 0xc000, 0x4400, 0x3400],
        [0xc000, 0x3400, 0x3800, 0x4400],
        [0x3000, 0x4400, 0xc000, 0xb800],
    ];
    let mut graph = Vec::new();
    for name in names {
        let cast_to_float = [text(1, "to"), integer(3, 1), integer(20, 2)].concat();
        let node = [
            text(1, &format!("{name}_half")),
            text(2, name),
            text(3, &format!("cast_{name}")),
            text(4, "Cast"),
            bytes(5, &cast_to_float),
        ]
        .concat();
        graph.extend(bytes(1, &node));
    }
    for (index, name) in names.into_iter().enumerate() {
        let input = if index == 0 {
            "input".to_owned()
        } else {
            format!("step{index}")
        };
        let output = if index == 2 {
            "output".to_owned()
        } else {
            format!("step{}", index + 1)
        };
        let node = [
            text(1, &input),
            text(1, name),
            text(2, &output),
            text(4, "Mul"),
        ]
        .concat();
        graph.extend(bytes(1, &node));
    }
    graph.extend(text(2, "stored_half_runtime_float"));
    for (name, values) in names.into_iter().zip(weights) {
        let raw = values
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let tensor = [
            integer(1, 1),
            integer(1, 4),
            integer(2, 10),
            text(8, &format!("{name}_half")),
            bytes(9, &raw),
        ]
        .concat();
        graph.extend(bytes(5, &tensor));
    }
    graph.extend(bytes(11, &value_info("input")));
    graph.extend(bytes(12, &value_info("output")));
    [
        integer(1, 8),
        text(2, "ferment-policy-regression"),
        bytes(7, &graph),
        bytes(8, &integer(2, 17)),
    ]
    .concat()
}

#[derive(Clone, Copy)]
enum Field<'a> {
    Integer(u64),
    Bytes(&'a [u8]),
}
fn take_varint(input: &mut &[u8]) -> u64 {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let (first, rest) = input.split_first().expect("complete protobuf varint");
        *input = rest;
        value |= u64::from(first & 127) << shift;
        if first & 128 == 0 {
            return value;
        }
    }
    panic!("invalid protobuf varint")
}
fn fields(mut input: &[u8]) -> Vec<(u64, Field<'_>)> {
    let mut out = Vec::new();
    while !input.is_empty() {
        let key = take_varint(&mut input);
        let value = match key & 7 {
            0 => Field::Integer(take_varint(&mut input)),
            wire @ (1 | 2 | 5) => {
                let length = match wire {
                    1 => 8,
                    5 => 4,
                    _ => take_varint(&mut input) as usize,
                };
                let (value, rest) = input
                    .split_at_checked(length)
                    .expect("complete protobuf field");
                input = rest;
                Field::Bytes(value)
            }
            wire => panic!("unsupported protobuf wire {wire}"),
        };
        out.push((key >> 3, value));
    }
    out
}
fn byte_fields(input: &[u8], number: u64) -> Vec<&[u8]> {
    fields(input)
        .into_iter()
        .filter_map(|(field, value)| match value {
            Field::Bytes(value) if field == number => Some(value),
            _ => None,
        })
        .collect()
}
fn integer_field(input: &[u8], number: u64) -> u64 {
    fields(input)
        .into_iter()
        .find_map(|(field, value)| match value {
            Field::Integer(value) if field == number => Some(value),
            _ => None,
        })
        .expect("required integer")
}
fn graph_shape(model: &[u8]) -> (usize, Vec<(u64, usize)>) {
    let graphs = byte_fields(model, 7);
    let graph = *graphs.first().expect("ONNX graph");
    assert_eq!(graphs.len(), 1);
    let casts = byte_fields(graph, 1)
        .into_iter()
        .filter(|node| byte_fields(node, 4) == [b"Cast".as_slice()])
        .count();
    let initializers = byte_fields(graph, 5)
        .into_iter()
        .map(|tensor| {
            (
                integer_field(tensor, 2),
                byte_fields(tensor, 9).iter().map(|data| data.len()).sum(),
            )
        })
        .collect();
    (casts, initializers)
}

fn optimized_graph_and_prediction(model: ModelId) -> (usize, Vec<(u64, usize)>) {
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "explicit pinned ORT runtime required"
    );
    let root = Directory::new();
    let input_path = root.0.join("input.onnx");
    let optimized_path = root.0.join("optimized.onnx");
    let original = model_bytes();
    assert_eq!(graph_shape(&original), (3, vec![(10, 8); 3]));
    std::fs::write(&input_path, &original).unwrap();
    let mut session = OrtBackend::session_builder(model, OrtExecutionProvider::Cpu, None)
        .unwrap()
        .with_optimized_model_path(&optimized_path)
        .unwrap()
        .commit_from_file(&input_path)
        .unwrap();
    for _ in 0..2 {
        let input =
            ort::value::Tensor::from_array(([1_i64, 4], vec![8_f32, -4., 2., -8.])).unwrap();
        let result = session.run(ort::inputs!["input" => input]).unwrap();
        let (shape, values) = result["output"].try_extract_tensor::<f32>().unwrap();
        assert_eq!(shape.to_vec(), vec![1_i64, 4]);
        assert_eq!(values, &[-1_f32, 8., -8., 4.]);
    }
    drop(session);
    assert_eq!(
        std::fs::read(&input_path).unwrap(),
        original,
        "approved input remains unchanged"
    );
    let optimized = std::fs::read(&optimized_path).unwrap();
    assert!(
        optimized.len() < 64 * 1024,
        "this test never loads the production-sized graph"
    );
    graph_shape(&optimized)
}

/// Kill: restore eager ConstantFolding for Parakeet; actual optimized graph then
/// loses the Casts and replaces the stored halves with resident FLOAT weights.
#[test]
#[ignore = "requires explicit pinned ORT runtime"]
fn parakeet_cpu_keeps_weight_casts_in_the_real_optimized_graph() {
    let (casts, tensors) = optimized_graph_and_prediction(ModelId::ParakeetEncoder);
    assert_eq!(
        casts, 3,
        "all original widening operations execute at runtime"
    );
    assert_eq!(tensors, vec![(10, 8); 3], "no eagerly widened initializer");
}

/// Kill: disable folding globally instead of only on the Parakeet CPU lane.
/// The same real graph must still fold on an unrelated model's CPU policy.
#[test]
#[ignore = "requires explicit pinned ORT runtime"]
fn other_cpu_models_retain_their_existing_constant_folding() {
    let (casts, tensors) = optimized_graph_and_prediction(ModelId::YuNetFace);
    assert_eq!(casts, 0, "unrelated policy keeps the original optimization");
    assert!(!tensors.is_empty());
    assert!(
        tensors.iter().all(|(dtype, _)| *dtype == 1),
        "folded weights are FLOAT"
    );
    assert!(tensors.iter().map(|(_, count)| count).sum::<usize>() >= 16);
}
