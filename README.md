# TensorRT-LLM Swift (Linux)

[![CI](https://github.com/wendylabsinc/tensorrt-llm-swift/actions/workflows/ci.yml/badge.svg)](https://github.com/wendylabsinc/tensorrt-llm-swift/actions/workflows/ci.yml)
![Swift 6.2+](https://img.shields.io/badge/Swift-6.2%2B-F05138?logo=swift&logoColor=white)
![Linux](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)
![TensorRT](https://img.shields.io/badge/TensorRT-10.x-76B900?logo=nvidia&logoColor=white)

Swift Package that provides Swift-first APIs for working with NVIDIA TensorRT-LLM on Linux.

This repository is **work in progress** and **subject to breaking changes** (including major public
API reshuffles) while the low-level foundations are still being established.

Swift 6.2 features are used aggressively where feasible:
- `InlineArray` to keep common small metadata (like shapes/strides) allocation-free.
- `Span` / `MutableSpan` / `Data.bytes` for safer, more composable views over contiguous memory at
  the boundaries where we hand buffers to TensorRT-LLM/CUDA.

## Requirements

- **Swift 6.2+** (the package is written in Swift 6 mode)
- **Linux** with TensorRT-LLM installed *or* a container environment where the TensorRT-LLM shared libraries
  are available at runtime (e.g. `libtensorrt_llm.so`, `libnvinfer.so`)
- For the end-to-end GPU test: a working NVIDIA driver stack accessible from the host/container
  (CUDA driver `libcuda.so` must be available)

Notes:
- This package currently targets **system-installed** TensorRT-LLM headers/libs (via a tiny C++ shim
  target that links `libnvinfer`, `libnvinfer_plugin`, `libnvonnxparser`, and `libtensorrt_llm` on Linux).
- You may need to ensure your container has access to the host GPU and driver libraries (e.g.
  NVIDIA Container Toolkit).

## What Works Today (System-Integrated APIs)

The following public APIs have real integration with the TensorRT-LLM system libraries (not stubs):

- `TensorRTLLMRuntimeProbe.inferRuntimeVersion()` (dynamic `dlopen` probe)
- `TensorRTLLMSystem.linkedRuntimeVersion()` (linked `libnvinfer` version)
- `TensorRTLLMSystem.buildIdentityEnginePlan(elementCount:)` (builds a tiny identity engine plan)
- `TensorRTLLMSystem.runIdentityPlanF32(plan:input:)` (runs the identity engine on GPU)
- `TensorRTLLMSystem.initializePlugins()` / `TensorRTLLMSystem.loadPluginLibrary(_:)` (plugin registration/loading)
- `TensorRTLLMRuntime.deserializeEngine(from:configuration:)` (deserializes and reflects IO surface)
- `TensorRTLLMRuntime.buildEngine(onnxURL:options:)` (builds a TensorRT plan via `nvonnxparser`)
- `ExecutionContext.enqueue(_:)` (executes a plan using host buffers)
- `ExecutionContext.enqueueDevice(inputs:outputs:synchronously:)` (device pointers + async support)
- `ExecutionQueue.external(streamIdentifier:)` (enqueue on a caller-provided CUDA stream)
- `ExecutionContext.recordEvent(_:)` + `TensorRTLLMSystem.CUDAEvent` (event-based completion)
- Dynamic shapes + profiles:
  - `ExecutionContext.reshape(bindings:)`
  - `ExecutionContext.setOptimizationProfile(named:)`
- Multi-GPU selection:
  - `DeviceSelection(gpu:)` is respected by `ExecutionContext`
- Convenience:
  - `ExecutionContext.enqueueF32(inputName:input:outputName:output:...)` (single-input/single-output)
  - `ExecutionContext.enqueueBytes(inputName:input:outputName:output:...)` (byte-level)

## Quick Start

### Add the package to your `Package.swift`

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MyApp",
    dependencies: [
        .package(url: "https://github.com/wendylabsinc/tensorrt-llm-swift", from: "0.0.1"),
    ],
    targets: [
        .executableTarget(
            name: "MyApp",
            dependencies: [
                .product(name: "TensorRTLLM", package: "tensorrt-llm-swift"),
            ]
        ),
    ]
)
```

### Probe TensorRT-LLM availability (dlopen)

```swift
import TensorRTLLM

let version = try TensorRTLLMRuntimeProbe.inferRuntimeVersion()
print("TensorRT-LLM runtime version: \(version)")
```

### Build an engine from ONNX (static shape) and run it

```swift
import TensorRTLLM

let runtime = TensorRTLLMRuntime()
let engine = try runtime.buildEngine(
    onnxURL: URL(fileURLWithPath: "model.onnx"),
    options: EngineBuildOptions(
        precision: [.fp32],
        workspaceSizeBytes: 1 << 28
    )
)

let ctx = try engine.makeExecutionContext()
let inputDesc = engine.description.inputs[0].descriptor

let input: [Float] = (0..<inputDesc.shape.elementCount).map(Float.init)
let inputBytes = input.withUnsafeBufferPointer { Data(buffer: $0) }

let batch = InferenceBatch(inputs: [
    inputDesc.name: TensorValue(descriptor: inputDesc, storage: .host(inputBytes))
])
let result = try await ctx.enqueue(batch)
```

### Build a dynamic ONNX engine with optimization profiles

If your ONNX model has dynamic shapes, you must provide optimization profiles at build time and
select a profile + `reshape(...)` at runtime.

```swift
import TensorRTLLM

let profile0 = OptimizationProfile(
    name: "0",
    axes: [:],
    bindingRanges: [
        "input": .init(min: TensorShape([1]), optimal: TensorShape([8]), max: TensorShape([16])),
    ]
)

let profile1 = OptimizationProfile(
    name: "1",
    axes: [:],
    bindingRanges: [
        "input": .init(min: TensorShape([32]), optimal: TensorShape([32]), max: TensorShape([64])),
    ]
)

let engine = try TensorRTLLMRuntime().buildEngine(
    onnxURL: URL(fileURLWithPath: "dynamic.onnx"),
    options: EngineBuildOptions(precision: [.fp32], profiles: [profile0, profile1])
)
let ctx = try engine.makeExecutionContext()
```

### Dynamic shapes checklist (profiles + reshape)

For dynamic-shape engines, the typical order is:

```swift
try await ctx.setOptimizationProfile(named: "0")
try await ctx.reshape(bindings: ["input": TensorShape([8])])
let result = try await ctx.enqueue(batch)
```

### Build, deserialize, and run a tiny identity engine (GPU)

```swift
import TensorRTLLM

// 1) Build a minimal engine plan using the system TensorRT builder.
let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: 8)

// 2) Deserialize the plan and inspect inputs/outputs (names + shapes).
let runtime = TensorRTLLMRuntime()
let engine = try runtime.deserializeEngine(from: plan)
print("Inputs:", engine.description.inputs.map(\.name))
print("Outputs:", engine.description.outputs.map(\.name))

// 3) Execute the plan via TensorRT enqueue + CUDA driver API.
let input: [Float] = (0..<8).map(Float.init)
let output = try TensorRTLLMSystem.runIdentityPlanF32(plan: plan, input: input)
precondition(output == input)
```

### Execute using the high-level `Engine` + `ExecutionContext`

```swift
import TensorRTLLM

let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: 8)
let engine = try TensorRTLLMRuntime().deserializeEngine(from: plan)
let ctx = try engine.makeExecutionContext()

let inputDesc = engine.description.inputs[0].descriptor
let outputDesc = engine.description.outputs[0].descriptor

let inputFloats: [Float] = (0..<8).map(Float.init)
let inputBytes = inputFloats.withUnsafeBufferPointer { Data(buffer: $0) }

let input = TensorValue(
    descriptor: inputDesc,
    storage: .host(inputBytes)
)

let batch = InferenceBatch(inputs: [inputDesc.name: input])
let result = try await ctx.enqueue(batch)
let outputValue = result.outputs[outputDesc.name]!
```

### Execute with caller-provided buffers (`[Float]` / `[UInt8]`)

```swift
import TensorRTLLM

let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: 8)
let engine = try TensorRTLLMRuntime().deserializeEngine(from: plan)
let ctx = try engine.makeExecutionContext()

let inputName = engine.description.inputs[0].name
let outputName = engine.description.outputs[0].name

let input: [Float] = (0..<8).map(Float.init)
var output: [Float] = []
try await ctx.enqueueF32(inputName: inputName, input: input, outputName: outputName, output: &output)
precondition(output == input)
```

## Examples

The package includes 17 examples organized by difficulty level. Run any example with `swift run <ExampleName>`.

### Beginner Examples

| Example | Description | Command |
|---------|-------------|---------|
| **HelloTensorRT** | Minimal "hello world" - probe version, build identity engine, run inference | `swift run HelloTensorRT` |
| **ONNXInference** | Load ONNX model, build engine, run inference with throughput measurement | `swift run ONNXInference` |
| **BatchProcessing** | Process multiple batches, latency statistics (p50/p95/p99) | `swift run BatchProcessing` |

### Intermediate Examples

| Example | Description | Command |
|---------|-------------|---------|
| **DynamicBatching** | Dynamic shapes for variable batch sizes at runtime | `swift run DynamicBatching` |
| **MultiProfile** | Multiple optimization profiles for different workloads | `swift run MultiProfile` |
| **AsyncInference** | Non-blocking inference with CUDA streams and events | `swift run AsyncInference` |
| **ImageClassifier** | End-to-end pipeline: preprocess → inference → postprocess | `swift run ImageClassifier` |
| **DeviceMemoryPipeline** | Keep tensors on GPU, avoid H2D/D2H transfers | `swift run DeviceMemoryPipeline` |

### Advanced Examples

| Example | Description | Command |
|---------|-------------|---------|
| **StreamingLLM** | Token-by-token generation with KV-cache pattern | `swift run StreamingLLM` |
| **MultiGPU** | Distribute inference across multiple GPUs | `swift run MultiGPU` |
| **CUDAEventPipelining** | Overlap compute with data transfer using events | `swift run CUDAEventPipelining` |
| **BenchmarkSuite** | Comprehensive throughput/latency measurement | `swift run BenchmarkSuite` |
| **FP16Quantization** | Compare FP32 vs FP16 precision and performance | `swift run FP16Quantization` |

### Real-World Examples

| Example | Description | Command |
|---------|-------------|---------|
| **TextEmbedding** | Sentence transformer for semantic search | `swift run TextEmbedding` |
| **ObjectDetection** | YOLO-style detection with NMS postprocessing | `swift run ObjectDetection` |
| **WhisperTranscription** | Audio transcription pipeline (encoder pattern) | `swift run WhisperTranscription` |
| **VisionTransformer** | ViT image classification with patch embeddings | `swift run VisionTransformer` |

### Example Output: BenchmarkSuite

```
=== TensorRT Benchmark Suite ===

┌──────────┬────────────┬────────────┬────────────┬────────────┐
│ Elements │ Throughput │ p50        │ p95        │ p99        │
├──────────┼────────────┼────────────┼────────────┼────────────┤
│ 64       │ 91.0K      │ 10.4 µs    │ 12.5 µs    │ 22.8 µs    │
│ 1024     │ 75.5K      │ 11.5 µs    │ 22.1 µs    │ 23.1 µs    │
│ 16384    │ 31.3K      │ 31.8 µs    │ 33.2 µs    │ 37.1 µs    │
└──────────┴────────────┴────────────┴────────────┴────────────┘
```

## Tests

Run:

```bash
swift test
```

The test suite includes end-to-end GPU tests that build engines (TensorRT builder and `nvonnxparser`),
deserialize them, and run inference (host buffers, device pointers, external streams, and CUDA events).
