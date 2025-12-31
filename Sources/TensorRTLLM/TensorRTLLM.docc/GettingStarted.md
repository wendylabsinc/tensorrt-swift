# Getting Started

This package targets **Swift 6.2+** and **Linux** machines where TensorRT-LLM and the NVIDIA driver stack
are available.

## Requirements

- Swift 6.2+
- Linux
- TensorRT-LLM system libraries available at runtime:
  - `libtensorrt_llm.so`
  - `libnvinfer.so`
  - `libnvinfer_plugin.so`
  - `libnvonnxparser.so` (only needed if you use ONNX build APIs)
- NVIDIA driver libraries available at runtime:
  - `libcuda.so` (CUDA Driver API)

## Quick smoke test

Probe that TensorRT-LLM can be loaded:

```swift
import TensorRTLLM

let version = try TensorRTLLMRuntimeProbe.inferRuntimeVersion()
print("TensorRT-LLM runtime version:", version)
```

Build and run a tiny identity engine (end-to-end GPU path):

```swift
import TensorRTLLM

let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: 8)
let input: [Float] = (0..<8).map(Float.init)
let output = try TensorRTLLMSystem.runIdentityPlanF32(plan: plan, input: input)
precondition(output == input)
```
