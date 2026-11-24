// The iOS side of the DINOv2 parity measurement.
//
// `dinov2_vits14_identity_fp32.mlmodelc` is the artifact the iOS engine ships
// (compiled by `Scripts/visual_semantic_encoder/convert.py`). This tool runs it
// on a raw 224x126 RGB frame and dumps the 384 floats, so the ONNX master's
// embedding can be compared against the real thing rather than against a
// re-implementation of it.
//
//   swiftc -O scripts/exports/dinov2/coreml_reference.swift -o /tmp/coreml_reference
//   /tmp/coreml_reference <model.mlmodelc> <frame.rgb> <out.f32>
//
// macOS only, and only useful where the iOS LocalModels tree is checked out.

import CoreML
import CoreVideo
import Foundation

let width = 224, height = 126

guard CommandLine.arguments.count == 4 else {
    FileHandle.standardError.write(
        Data("usage: coreml_reference <model.mlmodelc> <frame.rgb> <out.f32>\n".utf8))
    exit(2)
}
let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
let frame = [UInt8](try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])))
guard frame.count == width * height * 3 else {
    FileHandle.standardError.write(Data("frame must be \(width * height * 3) RGB bytes\n".utf8))
    exit(2)
}

// The model's ImageType is RGB; CoreML maps a 32BGRA buffer onto it.
var pixelBuffer: CVPixelBuffer?
guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                          [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
                          &pixelBuffer) == kCVReturnSuccess,
      let buffer = pixelBuffer
else {
    FileHandle.standardError.write(Data("CVPixelBufferCreate failed\n".utf8))
    exit(1)
}
CVPixelBufferLockBaseAddress(buffer, [])
let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
for y in 0..<height {
    for x in 0..<width {
        let source = (y * width + x) * 3
        let target = y * rowBytes + x * 4
        base[target] = frame[source + 2]      // B
        base[target + 1] = frame[source + 1]  // G
        base[target + 2] = frame[source]      // R
        base[target + 3] = 255
    }
}
CVPixelBufferUnlockBaseAddress(buffer, [])

let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuOnly  // deterministic; the ANE would add its own drift
let model = try MLModel(contentsOf: modelURL, configuration: configuration)
let input = try MLDictionaryFeatureProvider(dictionary: [
    "image": MLFeatureValue(pixelBuffer: buffer)
])
let output = try model.prediction(from: input)
guard let array = output.featureValue(for: "embedding")?.multiArrayValue else {
    FileHandle.standardError.write(Data("model produced no 'embedding'\n".utf8))
    exit(1)
}

var vector = [Float](repeating: 0, count: array.count)
for index in 0..<array.count { vector[index] = array[index].floatValue }
try Data(bytes: vector, count: vector.count * 4)
    .write(to: URL(fileURLWithPath: CommandLine.arguments[3]))
let norm = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
print("coreml embedding: \(vector.count) floats, norm \(norm)")
