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
  target that links `libnvinfer` on Linux).
- You may need to ensure your container has access to the host GPU and driver libraries (e.g.
  NVIDIA Container Toolkit).

## What Works Today (System-Integrated APIs)

The following public APIs have real integration with the TensorRT system libraries (not stubs):

- `TensorRTRuntimeProbe.inferRuntimeVersion()` (dynamic `dlopen` probe)
- `TensorRTSystem.linkedRuntimeVersion()` (linked `libnvinfer` version)
- `TensorRTSystem.buildIdentityEnginePlan(elementCount:)` (builds a tiny identity engine plan)
- `TensorRTSystem.runIdentityPlanF32(plan:input:)` (runs the identity engine on GPU)
- `TensorRTRuntime.deserializeEngine(from:configuration:)` (deserializes and reflects IO surface)
- `ExecutionContext.enqueue(_:)` (executes a plan using host buffers)
- `ExecutionContext.enqueueF32(inputName:input:outputName:output:...)` (single-input/single-output convenience)

## Quick Start

### Add the package to your `Package.swift`

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MyApp",
    dependencies: [
        .package(url: "https://github.com/wendylabsinc/tensorrt-swift", .branch("main")),
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

### Probe TensorRT availability (dlopen)

```swift
import TensorRT

let version = try TensorRTRuntimeProbe.inferRuntimeVersion()
print("TensorRT runtime version: \(version)")
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

The test suite includes an end-to-end GPU test that builds an identity engine with TensorRT,
deserializes it, and runs inference.
