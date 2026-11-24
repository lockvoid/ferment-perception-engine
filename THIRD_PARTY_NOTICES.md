# Third-party notices

- The audio and video minimaps embed Noto Sans Regular and SemiBold, version 2.015, under the SIL Open Font License 1.1. [License](crates/engine-processors/src/resources/fonts/OFL.txt) · [Source and hashes](crates/engine-processors/src/resources/fonts/provenance.json)
- The saliency fixtures use resized copies of the OpenCV YuNet demo image. The generator pins the original image by SHA-256. [Upstream demo](https://huggingface.co/opencv/face_detection_yunet/blob/main/example_outputs/largest_selfie.jpg) · [MIT license](scripts/exports/video_saliency/LICENSE-YUNET)
- The vendored Core ML bindings retain their upstream Apache 2.0 and MIT license files. [Apache 2.0](vendor/coreml-native/LICENSE-APACHE) · [MIT](vendor/coreml-native/LICENSE-MIT)
- The ONNX Runtime C headers in the Windows ML adapter retain their Microsoft copyright and MIT terms. [MIT license](vendor/winml-native/LICENSE-ONNX-RUNTIME)
- Adapted FluidAudio and Pillow code has its own provenance and notices. [FluidAudio](crates/engine-processors/notices/THIRD_PARTY_NOTICES.md) · [Pillow](crates/engine-retrieval/THIRD_PARTY_NOTICES.md)
- The FastDVDnet export uses the vendored MIT-licensed architecture. [Upstream license](scripts/exports/fastdvdnet/vendor/LICENSE)

The library uses a dynamically linked LGPL build of FFmpeg. The provisioner records the upstream source and build inputs. Model weights and external test vectors have their own terms and are obtained separately. [FFmpeg license](https://ffmpeg.org/legal.html) · [Provisioning recipe](scripts/provision-ffmpeg)
