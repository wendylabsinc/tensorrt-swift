# TensorRT Swift (Linux)

[![CI](https://github.com/wendylabsinc/tensorrt-swift/actions/workflows/ci.yml/badge.svg)](https://github.com/wendylabsinc/tensorrt-swift/actions/workflows/ci.yml)
![Swift 6.2+](https://img.shields.io/badge/Swift-6.2%2B-F05138?logo=swift&logoColor=white)
![Linux](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)
![TensorRT](https://img.shields.io/badge/TensorRT-10.x-76B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)

Swift Package that provides Swift-first APIs for working with NVIDIA TensorRT on Linux, with a separate TensorRTLLM product for LLM-specific extensions.

> **Note**: The `TensorRT` product wraps the **TensorRT** inference engine. The `TensorRTLLM` product is a thin extension layer today; full TensorRT-LLM integration (in-flight batching, KV-cache management, tensor parallelism) is planned for future releases.

This repository is **work in progress** and **subject to breaking changes** while the low-level foundations are being established.

Swift 6.2 features are used aggressively where feasible:
- `InlineArray` to keep common small metadata (like shapes/strides) allocation-free
- `Span` / `MutableSpan` / `Data.bytes` for safer, more composable views over contiguous memory
- Actor-based `ExecutionContext` for thread-safe inference

## System Requirements

### Required Libraries

The package links against the following system libraries at **build time** and **runtime**:

| Library | Package | Purpose |
|---------|---------|---------|
| `libnvinfer.so` | TensorRT | Core inference engine |
| `libnvinfer_plugin.so` | TensorRT | Built-in plugins |
| `libnvonnxparser.so` | TensorRT | ONNX model import |
| `libcuda.so` | CUDA Driver | GPU access |

### Installation

#### Option 1: NVIDIA Container (Recommended)

Use the official TensorRT container which includes all dependencies:

```bash
docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.08-py3
```

#### Option 1b: Jetson Container (Orin Nano, AGX Thor)

Jetson uses aarch64 containers and must match the host JetPack/L4T release. See
`docs/jetson-container.md` for a full recipe.

#### Option 2: System Installation (Ubuntu/Debian)

```bash
# 1. Install CUDA 12.6
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

# 2. Install TensorRT 10.x
sudo apt-get install -y libnvinfer10 libnvinfer-plugin10 libnvonnxparser10 libnvinfer-dev

# 3. Add CUDA to your path
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

#### Option 3: From NVIDIA Developer Downloads

1. Download [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads)
2. Download [TensorRT 10.x](https://developer.nvidia.com/tensorrt) (requires NVIDIA Developer account)
3. Follow NVIDIA's installation guides

### Verifying Installation

```bash
# Check CUDA
nvcc --version

# Check TensorRT
dpkg -l | grep nvinfer
# or
ls /usr/lib/x86_64-linux-gnu/libnvinfer*
```

### Swift Installation

Install Swift 6.2+ via [Swiftly](https://swiftlang.github.io/swiftly/):

```bash
curl -L https://swiftlang.github.io/swiftly/swiftly-install.sh | bash
swiftly install 6.2
```

## What Works Today

### Core APIs

| API | Description |
|-----|-------------|
| `TensorRTRuntime.buildEngine(onnxURL:options:)` | Build TensorRT engine from ONNX |
| `TensorRTRuntime.deserializeEngine(from:)` | Load serialized engine plan |
| `Engine.save(to:)` / `Engine.load(from:)` | Persist/load engines to disk |
| `ExecutionContext.enqueue(_:)` | Execute inference (host buffers) |
| `ExecutionContext.enqueueDevice(...)` | Execute with device pointers |
| `ExecutionContext.warmup(iterations:)` | Warmup for stable latency |

### GPU & Device APIs

| API | Description |
|-----|-------------|
| `TensorRTSystem.cudaDeviceCount()` | Number of available GPUs |
| `TensorRTSystem.deviceProperties(device:)` | GPU name, compute capability, memory |
| `TensorRTSystem.memoryInfo(device:)` | Free/total GPU memory |
| `TensorRTSystem.CUDAStream` | RAII stream wrapper |
| `TensorRTSystem.CUDAEvent` | RAII event wrapper |

### Dynamic Shapes & Profiles

| API | Description |
|-----|-------------|
| `ExecutionContext.reshape(bindings:)` | Set input shapes at runtime |
| `ExecutionContext.setOptimizationProfile(named:)` | Switch optimization profiles |
| `OptimizationProfile` | Define min/opt/max shapes |

### LLM Extensions (TensorRTLLM)

| API | Description |
|-----|-------------|
| `ExecutionContext.stream(...)` | Streaming inference (AsyncSequence) |
| `StreamingConfiguration` | Configure token-by-token generation |
| `StreamingInferenceStep` | Per-step metadata and outputs |

### Swift-y Conveniences

```swift
// TensorShape with array literal
let shape: TensorShape = [1, 3, 224, 224]
print(shape)        // "TensorShape[1, 3, 224, 224]"
print(shape[0])     // 1

// Engine persistence
try engine.save(to: URL(fileURLWithPath: "model.engine"))
let loaded = try Engine.load(from: URL(fileURLWithPath: "model.engine"))

// Query GPU before loading
let mem = try TensorRTSystem.memoryInfo()
print("Free GPU memory: \(mem.free / 1_000_000_000) GB")
```

## Quick Start

### Add the package to your `Package.swift`

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MyApp",
    dependencies: [
        .package(url: "https://github.com/wendylabsinc/tensorrt-swift", from: "0.0.1"),
    ],
    targets: [
        .executableTarget(
            name: "MyApp",
            dependencies: [
                .product(name: "TensorRT", package: "tensorrt-swift"),
            ]
        ),
    ]
)
```

To use the LLM extension module for streaming inference and other LLM utilities:

```swift
.product(name: "TensorRTLLM", package: "tensorrt-swift")
```

### Query GPU and TensorRT version

```swift
import TensorRT
// Check TensorRT version
let version = try TensorRTRuntimeProbe.inferRuntimeVersion()
print("TensorRT version: \(version)")

// Check GPU
let props = try TensorRTSystem.deviceProperties()
print("GPU: \(props.name)")
print("Compute Capability: \(props.computeCapability)")
print("Memory: \(props.totalMemory / 1_000_000_000) GB")

let mem = try TensorRTSystem.memoryInfo()
print("Free: \(mem.free / 1_000_000_000) GB / \(mem.total / 1_000_000_000) GB")
```

### Build an engine from ONNX and run inference

```swift
import TensorRT
let runtime = TensorRTRuntime()
let engine = try runtime.buildEngine(
    onnxURL: URL(fileURLWithPath: "model.onnx"),
    options: EngineBuildOptions(
        precision: [.fp32],
        workspaceSizeBytes: 1 << 28
    )
)

// Save for later use (avoid rebuild)
try engine.save(to: URL(fileURLWithPath: "model.engine"))

let ctx = try engine.makeExecutionContext()

// Warmup for stable latency
let warmup = try await ctx.warmup(iterations: 10)
print("Warmup avg: \(warmup.average ?? .zero)")

// Run inference
let inputDesc = engine.description.inputs[0].descriptor
let input: [Float] = (0..<inputDesc.shape.elementCount).map(Float.init)
let inputBytes = input.withUnsafeBufferPointer { Data(buffer: $0) }

let batch = InferenceBatch(inputs: [
    inputDesc.name: TensorValue(descriptor: inputDesc, storage: .host(inputBytes))
])
let result = try await ctx.enqueue(batch)
```

### Streaming inference (for LLMs)

```swift
import TensorRTLLM
let stream = context.stream(
    initialBatch: promptBatch,
    configuration: StreamingConfiguration(maxSteps: 100)
) { previousResult in
    // Transform previous output into next input (e.g., append generated token)
    return makeNextBatch(from: previousResult)
}

for try await step in stream {
    print("Step \(step.stepIndex), final: \(step.isFinal)")
    // Process each step as it arrives
    if step.isFinal { break }
}
```

### Dynamic shapes with optimization profiles

```swift
import TensorRT
let profile = OptimizationProfile(
    name: "batch_range",
    axes: [:],
    bindingRanges: [
        "input": .init(
            min: TensorShape([1, 512]),
            optimal: TensorShape([8, 512]),
            max: TensorShape([32, 512])
        ),
    ]
)

let engine = try TensorRTRuntime().buildEngine(
    onnxURL: URL(fileURLWithPath: "dynamic.onnx"),
    options: EngineBuildOptions(precision: [.fp32], profiles: [profile])
)

let ctx = try engine.makeExecutionContext()
try await ctx.reshape(bindings: ["input": TensorShape([16, 512])])
let result = try await ctx.enqueue(batch)
```

## Examples

The package includes 17 examples organized by difficulty level. Run any example with `./scripts/swiftw run <ExampleName>`.
The wrapper keeps build artifacts in `/tmp` by default; override with `SWIFT_BUILD_PATH` if needed.

### Beginner Examples

| Example | Description | Command |
|---------|-------------|---------|
| **HelloTensorRT** | Minimal "hello world" - probe version, build identity engine, run inference | `./scripts/swiftw run HelloTensorRT` |
| **ONNXInference** | Load ONNX model, build engine, run inference with throughput measurement | `./scripts/swiftw run ONNXInference` |
| **BatchProcessing** | Process multiple batches, latency statistics (p50/p95/p99) | `./scripts/swiftw run BatchProcessing` |

### Intermediate Examples

| Example | Description | Command |
|---------|-------------|---------|
| **DynamicBatching** | Dynamic shapes for variable batch sizes at runtime | `./scripts/swiftw run DynamicBatching` |
| **MultiProfile** | Multiple optimization profiles for different workloads | `./scripts/swiftw run MultiProfile` |
| **AsyncInference** | Non-blocking inference with CUDA streams and events | `./scripts/swiftw run AsyncInference` |
| **ImageClassifier** | End-to-end pipeline: preprocess → inference → postprocess | `./scripts/swiftw run ImageClassifier` |
| **DeviceMemoryPipeline** | Keep tensors on GPU, avoid H2D/D2H transfers | `./scripts/swiftw run DeviceMemoryPipeline` |

### Advanced Examples

| Example | Description | Command |
|---------|-------------|---------|
| **StreamingLLM** | Token-by-token generation with KV-cache pattern | `./scripts/swiftw run StreamingLLM` |
| **MultiGPU** | Distribute inference across multiple GPUs | `./scripts/swiftw run MultiGPU` |
| **CUDAEventPipelining** | Overlap compute with data transfer using events | `./scripts/swiftw run CUDAEventPipelining` |
| **BenchmarkSuite** | Comprehensive throughput/latency measurement | `./scripts/swiftw run BenchmarkSuite` |
| **FP16Quantization** | Compare FP32 vs FP16 precision and performance | `./scripts/swiftw run FP16Quantization` |

### Real-World Examples

| Example | Description | Command |
|---------|-------------|---------|
| **TextEmbedding** | Sentence transformer for semantic search | `./scripts/swiftw run TextEmbedding` |
| **ObjectDetection** | YOLO-style detection with NMS postprocessing | `./scripts/swiftw run ObjectDetection` |
| **WhisperTranscription** | Audio transcription pipeline (encoder pattern) | `./scripts/swiftw run WhisperTranscription` |
| **VisionTransformer** | ViT image classification with patch embeddings | `./scripts/swiftw run VisionTransformer` |

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
./scripts/swiftw test
```

This wrapper keeps build artifacts in `/tmp` by default to avoid `.build` permission issues. Override with
`SWIFT_BUILD_PATH=/your/path ./scripts/swiftw test` if needed.

The test suite includes end-to-end GPU tests that build engines (TensorRT builder and `nvonnxparser`),
deserialize them, and run inference (host buffers, device pointers, external streams, and CUDA events).

## Troubleshooting

### `libnvinfer.so: cannot open shared object file`

TensorRT libraries are not in your library path. Add them:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# or wherever TensorRT is installed
```

### `CUDA driver version is insufficient`

Your NVIDIA driver is too old for CUDA 12.6. Update your driver:

```bash
sudo apt-get install nvidia-driver-550  # or newer
```

### Swift can't find CUDA headers

Ensure CUDA is installed and the include path is correct:

```bash
ls /usr/local/cuda/include/cuda.h
# If not found, create symlink or adjust Package.swift
```

## License

See [LICENSE.txt](LICENSE.txt).
