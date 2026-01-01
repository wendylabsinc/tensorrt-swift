# Building From ONNX

`TensorRTRuntime` can build a TensorRT engine directly from an ONNX file using the system
`nvonnxparser` library:

```swift
import TensorRT
let engine = try TensorRTRuntime().buildEngine(
    onnxURL: URL(fileURLWithPath: "model.onnx"),
    options: EngineBuildOptions(
        precision: [.fp32],
        workspaceSizeBytes: 1 << 28
    )
)
```

The returned ``Engine`` includes:
- `serialized`: the built plan bytes (so you can persist/cache it if you want)
- `description`: reflected input/output bindings

## Static vs dynamic shapes

- For **static** ONNX models (all concrete dimensions), no profiles are required.
- For **dynamic** ONNX models (any `-1`/symbolic dimensions), you must supply explicit optimization
  profiles via ``EngineBuildOptions/profiles`` or provide fixed-shape hints via
  ``EngineBuildOptions/shapeHints``. See <doc:DynamicShapesAndProfiles>.

If you only need a single fixed shape for a dynamic model, shape hints are the simplest option:

```swift
let options = EngineBuildOptions(
    precision: [.fp16],
    shapeHints: ["input": TensorShape([1, 3, 224, 224])]
)
let engine = try TensorRTRuntime().buildEngine(onnxURL: modelURL, options: options)
```
