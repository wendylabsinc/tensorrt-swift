# ``TensorRTLLM``

Swift-first APIs for NVIDIA TensorRT-LLM on Linux (Swift 6.2+).

This package is a **work in progress** and may introduce breaking changes frequently while the
core runtime/building surface stabilizes.

## Topics

### Runtime

- ``TensorRTLLMRuntime``
- ``Engine``
- ``ExecutionContext``

### Building Engines

- ``EngineBuildOptions``
- ``OptimizationProfile``
- ``TensorRTLLMRuntime/buildEngine(onnxURL:options:)``

### Loading Engines

- ``EngineLoadConfiguration``
- ``TensorRTLLMRuntime/deserializeEngine(from:configuration:)``

### Dynamic Shapes & Profiles

- ``ExecutionContext/reshape(bindings:)``
- ``ExecutionContext/setOptimizationProfile(named:)``

### System Integration

- ``TensorRTLLMSystem``
- ``TensorRTLLMRuntimeProbe``

## Articles

- <doc:GettingStarted>
- <doc:BuildingFromONNX>
- <doc:DynamicShapesAndProfiles>
- <doc:Plugins>
- <doc:ExecutionModels>
