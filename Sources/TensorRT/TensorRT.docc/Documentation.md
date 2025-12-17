# ``TensorRT``

Swift-first APIs for NVIDIA TensorRT on Linux (Swift 6.2+).

This package is a **work in progress** and may introduce breaking changes frequently while the
core runtime/building surface stabilizes.

## Topics

### Runtime

- ``TensorRTRuntime``
- ``Engine``
- ``ExecutionContext``

### Building Engines

- ``EngineBuildOptions``
- ``OptimizationProfile``
- ``TensorRTRuntime/buildEngine(onnxURL:options:)``

### Loading Engines

- ``EngineLoadConfiguration``
- ``TensorRTRuntime/deserializeEngine(from:configuration:)``

### Dynamic Shapes & Profiles

- ``ExecutionContext/reshape(bindings:)``
- ``ExecutionContext/setOptimizationProfile(named:)``

### System Integration

- ``TensorRTSystem``
- ``TensorRTRuntimeProbe``

## Articles

- <doc:GettingStarted>
- <doc:BuildingFromONNX>
- <doc:DynamicShapesAndProfiles>
- <doc:Plugins>
- <doc:ExecutionModels>
