# TensorRT Swift (Linux)

Swift Package that provides Swift-first APIs for working with NVIDIA TensorRT engines on Linux.

This repository is a **work in progress**. The public API is still evolving and not all planned
features are implemented yet.

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

## Quick Start

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

## Tests

Run:

```bash
swift test
```

The test suite includes an end-to-end GPU test that builds an identity engine with TensorRT,
deserializes it, and runs inference.

