# TensorRT Swift (Linux)

![Swift 6.2+](https://img.shields.io/badge/Swift-6.2%2B-orange)
![Linux](https://img.shields.io/badge/Platform-Linux-blue)

Swift Package that provides Swift-first APIs for working with NVIDIA TensorRT engines on Linux.

This repository is **work in progress** and **subject to breaking changes** (including major public
API reshuffles) while the low-level foundations are still being established.

Swift 6.2 features are used aggressively where feasible:
- `InlineArray` to keep common small metadata (like shapes/strides) allocation-free.
- `Span` / `MutableSpan` / `Data.bytes` for safer, more composable views over contiguous memory at
  the boundaries where we hand buffers to TensorRT/CUDA.

## Requirements

- **Swift 6.2+** (the package is written in Swift 6 mode)
- **Linux** with TensorRT installed *or* a container environment where the TensorRT shared libraries
  are available at runtime (e.g. `libnvinfer.so`)
- For the end-to-end GPU test: a working NVIDIA driver stack accessible from the host/container
  (CUDA driver `libcuda.so` must be available)

Notes:
- This package currently targets **system-installed** TensorRT headers/libs (via a tiny C++ shim
  target that links `libnvinfer`, `libnvinfer_plugin`, and `libnvonnxparser` on Linux).
- You may need to ensure your container has access to the host GPU and driver libraries (e.g.
  NVIDIA Container Toolkit).

## What Works Today (System-Integrated APIs)

The following public APIs have real integration with the TensorRT system libraries (not stubs):

- `TensorRTRuntimeProbe.inferRuntimeVersion()` (dynamic `dlopen` probe)
- `TensorRTSystem.linkedRuntimeVersion()` (linked `libnvinfer` version)
- `TensorRTSystem.buildIdentityEnginePlan(elementCount:)` (builds a tiny identity engine plan)
- `TensorRTSystem.runIdentityPlanF32(plan:input:)` (runs the identity engine on GPU)
- `TensorRTSystem.initializePlugins()` / `TensorRTSystem.loadPluginLibrary(_:)` (plugin registration/loading)
- `TensorRTRuntime.deserializeEngine(from:configuration:)` (deserializes and reflects IO surface)
- `TensorRTRuntime.buildEngine(onnxURL:options:)` (builds a TensorRT plan via `nvonnxparser`)
- `ExecutionContext.enqueue(_:)` (executes a plan using host buffers)
- `ExecutionContext.enqueueDevice(inputs:outputs:synchronously:)` (device pointers + async support)
- `ExecutionQueue.external(streamIdentifier:)` (enqueue on a caller-provided CUDA stream)
- `ExecutionContext.recordEvent(_:)` + `TensorRTSystem.CUDAEvent` (event-based completion)
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

To track the latest (breaking) changes during development, you can also depend on the `main` branch:

```swift
.package(url: "https://github.com/wendylabsinc/tensorrt-swift", .branch("main"))
```

### Probe TensorRT availability (dlopen)

```swift
import TensorRT

let version = try TensorRTRuntimeProbe.inferRuntimeVersion()
print("TensorRT runtime version: \(version)")
```

### Build an engine from ONNX (static shape) and run it

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
import TensorRT

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

let engine = try TensorRTRuntime().buildEngine(
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
import TensorRT

// 1) Build a minimal engine plan using the system TensorRT builder.
let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)

// 2) Deserialize the plan and inspect inputs/outputs (names + shapes).
let runtime = TensorRTRuntime()
let engine = try runtime.deserializeEngine(from: plan)
print("Inputs:", engine.description.inputs.map(\.name))
print("Outputs:", engine.description.outputs.map(\.name))

// 3) Execute the plan via TensorRT enqueue + CUDA driver API.
let input: [Float] = (0..<8).map(Float.init)
let output = try TensorRTSystem.runIdentityPlanF32(plan: plan, input: input)
precondition(output == input)
```

### Execute using the high-level `Engine` + `ExecutionContext`

```swift
import TensorRT

let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)
let engine = try TensorRTRuntime().deserializeEngine(from: plan)
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
import TensorRT

let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)
let engine = try TensorRTRuntime().deserializeEngine(from: plan)
let ctx = try engine.makeExecutionContext()

let inputName = engine.description.inputs[0].name
let outputName = engine.description.outputs[0].name

let input: [Float] = (0..<8).map(Float.init)
var output: [Float] = []
try await ctx.enqueueF32(inputName: inputName, input: input, outputName: outputName, output: &output)
precondition(output == input)
```

## Tests

Run:

```bash
swift test
```

The test suite includes end-to-end GPU tests that build engines (TensorRT builder and `nvonnxparser`),
deserialize them, and run inference (host buffers, device pointers, external streams, and CUDA events).
