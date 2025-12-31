# Dynamic Shapes and Optimization Profiles

TensorRT-LLM requires optimization profiles for engines with dynamic shapes.

This package models profiles using ``OptimizationProfile`` and attaches them at build time via
``EngineBuildOptions``.

## Build with profiles (ONNX)

```swift
import TensorRTLLM

let p0 = OptimizationProfile(
    name: "0",
    axes: [:],
    bindingRanges: [
        "input": .init(min: TensorShape([1]), optimal: TensorShape([8]), max: TensorShape([16])),
    ]
)

let p1 = OptimizationProfile(
    name: "1",
    axes: [:],
    bindingRanges: [
        "input": .init(min: TensorShape([32]), optimal: TensorShape([32]), max: TensorShape([64])),
    ]
)

let engine = try TensorRTLLMRuntime().buildEngine(
    onnxURL: URL(fileURLWithPath: "dynamic.onnx"),
    options: EngineBuildOptions(precision: [.fp32], profiles: [p0, p1])
)
```

## Select a profile at runtime

Before enqueueing inference for a given shape:

1. Select the desired profile with ``ExecutionContext/setOptimizationProfile(named:)``.
2. Provide concrete input shapes with ``ExecutionContext/reshape(bindings:)``.
3. Call ``ExecutionContext/enqueue(_:synchronously:)``.

```swift
let ctx = try engine.makeExecutionContext()

try await ctx.setOptimizationProfile(named: "0")
try await ctx.reshape(bindings: ["input": TensorShape([8])])
let result = try await ctx.enqueue(batch)
```

## Notes

- Profile switching is per-context state. If you use multiple contexts concurrently, each context
  selects its own active profile.
- If you use ``ExecutionQueue/external(streamIdentifier:)``, the external stream must belong to the
  same CUDA device/context as the engine/context.
