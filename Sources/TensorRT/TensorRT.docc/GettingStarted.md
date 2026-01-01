# Getting Started

This package targets **Swift 6.2+** and **Linux** machines where TensorRT and the NVIDIA driver stack
are available.

## Requirements

- Swift 6.2+
- Linux
- TensorRT system libraries available at runtime:
  - `libnvinfer.so`
  - `libnvinfer_plugin.so`
  - `libnvonnxparser.so` (only needed if you use ONNX build APIs)
- NVIDIA driver libraries available at runtime:
  - `libcuda.so` (CUDA Driver API)

## Quick smoke test

Probe that TensorRT can be loaded:

```swift
import TensorRT
let version = try TensorRTRuntimeProbe.inferRuntimeVersion()
print("TensorRT runtime version:", version)
```

Build and run a tiny identity engine (end-to-end GPU path):

```swift
import TensorRT
let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)
let input: [Float] = (0..<8).map(Float.init)
let output = try TensorRTSystem.runIdentityPlanF32(plan: plan, input: input)
precondition(output == input)
```
