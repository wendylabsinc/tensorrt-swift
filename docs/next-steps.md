# Next Steps for Linux (Kubuntu, NVIDIA GPU)

Target environment: Swift 6.2+, Kubuntu on an NVIDIA GPU (e.g., RTX 4070 Ti), TensorRT/CUDA installed.

## 1) Wire the native interface (Swift 6.2 C++ interop)
- Implement `DefaultTensorRTNativeInterface` in `Sources/TensorRT/Interop/` using Swift 6.2 C++ interop and the Span/InlineArray proposals (see `docs/swift62-features/`).
- Map APIs:
  - Deserialize/build engines (`deserializeEngine`, `buildEngine` with `NetworkDefinition`, `buildEngine(onnxURL:)`).
  - Create execution contexts, set profiles, reshape, enqueue, warmup.
  - Allocate persistent buffers using CUDA pinned/managed/device allocations.
  - Surface handles (`EngineHandle`) for context creation and cleanup.
- Use `Span/MutableSpan/OutputSpan` for host/device buffers and calibration caches; use InlineArray for small shape/stride metadata.

## 2) Bring in TensorRT/CUDA headers and link
- Install CUDA and TensorRT (matching driver) on Kubuntu.
- Add a module map or C++ shim target that exposes the TensorRT C++ API to Swift (guard with `#if canImport(TensorRTCxx)`).
- In `Package.swift`, add the interop target (e.g., `TensorRTNative`) that links against TensorRT/CUDA libs and is only enabled on Linux.
- Ensure RPATH/library search paths point to TensorRT/CUDA (typically `/usr/lib/x86_64-linux-gnu` or the TensorRT tarball locations).

## 3) Shape/weight handling and validation
- Extend `NetworkBuilder` to carry weights/biases per layer (prefer spans/inline arrays to avoid copies).
- Add shape inference and validation for the added layers; emit `NetworkValidationIssue` entries instead of throwing early where possible.
- Implement per-binding quantization metadata and refit/calibration flows (INT8/INT4/BF16).

## 4) Persistent buffers and scheduling
- Wire `PersistentBufferPlan` to real CUDA allocations (pinned/managed/device) and integrate pooled buffers with `TensorValue.Storage.pooledBuffer`.
- Implement `InferenceScheduler` policies (latency vs throughput) and stream/graph selection in `ExecutionQueue`.

## 5) Testing strategy on Kubuntu + GPU
- Unit tests: keep using `MockExecutionContext` for API-level tests without TensorRT.
- Integration tests (GPU): add a small ONNX model fixture, build an engine, run `enqueue`, and verify outputs; guard with `#if canImport(TensorRTCxx)` and a feature flag to skip on non-GPU runners.
- Build check: `swift build` on Linux with TensorRT present; optionally cross-check with `swift build --triple x86_64-unknown-linux-gnu` on mac just for compile failures (no linking).

## 6) Developer ergonomics
- Add doc snippets showing how to select precision/profiles and how to map CUDA stream handles into `ExecutionQueue`.
- Document environment variables or SwiftPM settings for library paths and TensorRT version selection.
- Keep mac builds working by wrapping native imports with `#if canImport(TensorRTCxx)` and defaulting to mocks when unavailable.
