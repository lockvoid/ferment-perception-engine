// The vImage side of the resampler parity gate.
//
// iOS prepares the encoder's 224x126 input with
// `vImageScale_ARGB8888(..., kvImageHighQualityResampling)` — a Lanczos-class
// kernel whose exact taps Apple does not document, so the Rust port cannot be
// bit-faithful. This tool writes the two files that turn that unknown into a
// measured, pinned number:
//
//   goldens/resample_source_640x360.rgb   the 640-wide ai-feed frame, packed RGB
//   goldens/resample_vimage_224x126.rgb   what vImage makes of it
//
// `crates/engine-processors/src/dinov2/mod.rs` reads both and holds the Rust
// Lanczos3 resample to the drift recorded there; `measure_drift.py` runs the two
// resamples through the ONNX graph and reports the embedding cosine distance.
//
//   swiftc -O scripts/exports/dinov2/vimage_reference.swift -o /tmp/vimage_reference
//   /tmp/vimage_reference scripts/exports/dinov2/goldens
//
// Run once; the outputs are committed. macOS only, by construction.

import Accelerate
import Foundation

let sourceWidth = 640, sourceHeight = 360
let targetWidth = 224, targetHeight = 126

/// A frame with everything a resampler can trip over: a smooth two-axis ramp,
/// a hard-edged block grid, and a band of seeded value noise.
func makeSource() -> [UInt8] {
    var rgb = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 3)
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    func next() -> UInt8 {
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return UInt8(truncatingIfNeeded: (state &* 0x2545_F491_4F6C_DD1D) >> 56)
    }
    for y in 0..<sourceHeight {
        for x in 0..<sourceWidth {
            let index = (y * sourceWidth + x) * 3
            if y < sourceHeight / 3 {
                // Smooth ramp.
                rgb[index] = UInt8(x * 255 / (sourceWidth - 1))
                rgb[index + 1] = UInt8(y * 255 / (sourceHeight - 1))
                rgb[index + 2] = UInt8((x + y) * 255 / (sourceWidth + sourceHeight - 2))
            } else if y < 2 * sourceHeight / 3 {
                // Hard edges: a 16 px checkerboard with saturated corners.
                let on = ((x / 16) + (y / 16)) % 2 == 0
                rgb[index] = on ? 255 : 0
                rgb[index + 1] = on ? 16 : 240
                rgb[index + 2] = on ? 128 : 64
            } else {
                // High-frequency noise — the worst case for any kernel.
                rgb[index] = next()
                rgb[index + 1] = next()
                rgb[index + 2] = next()
            }
        }
    }
    return rgb
}

/// vImage scales four interleaved 8-bit channels; the alpha lane is padding
/// here and is dropped again on the way out. Channels are independent under a
/// separable kernel, so scaling RGBX is exactly scaling RGB.
func vImageScaled(_ rgb: [UInt8]) -> [UInt8] {
    var source = [UInt8](repeating: 255, count: sourceWidth * sourceHeight * 4)
    for pixel in 0..<(sourceWidth * sourceHeight) {
        source[pixel * 4] = rgb[pixel * 3]
        source[pixel * 4 + 1] = rgb[pixel * 3 + 1]
        source[pixel * 4 + 2] = rgb[pixel * 3 + 2]
    }
    var target = [UInt8](repeating: 0, count: targetWidth * targetHeight * 4)

    let ok: Bool = source.withUnsafeMutableBytes { src in
        target.withUnsafeMutableBytes { dst in
            var sourceBuffer = vImage_Buffer(
                data: src.baseAddress, height: vImagePixelCount(sourceHeight),
                width: vImagePixelCount(sourceWidth), rowBytes: sourceWidth * 4)
            var targetBuffer = vImage_Buffer(
                data: dst.baseAddress, height: vImagePixelCount(targetHeight),
                width: vImagePixelCount(targetWidth), rowBytes: targetWidth * 4)
            return vImageScale_ARGB8888(
                &sourceBuffer, &targetBuffer, nil,
                vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError
        }
    }
    guard ok else {
        FileHandle.standardError.write(Data("vImageScale_ARGB8888 failed\n".utf8))
        exit(1)
    }

    var out = [UInt8](repeating: 0, count: targetWidth * targetHeight * 3)
    for pixel in 0..<(targetWidth * targetHeight) {
        out[pixel * 3] = target[pixel * 4]
        out[pixel * 3 + 1] = target[pixel * 4 + 1]
        out[pixel * 3 + 2] = target[pixel * 4 + 2]
    }
    return out
}

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: vimage_reference <goldens-dir>\n".utf8))
    exit(2)
}
let directory = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

let source = makeSource()
let scaled = vImageScaled(source)
try Data(source).write(to: directory.appendingPathComponent("resample_source_640x360.rgb"))
try Data(scaled).write(to: directory.appendingPathComponent("resample_vimage_224x126.rgb"))
print("wrote \(source.count) source bytes and \(scaled.count) vImage bytes to \(directory.path)")
