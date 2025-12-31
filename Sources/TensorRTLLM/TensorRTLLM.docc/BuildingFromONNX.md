# Building From ONNX

`TensorRTLLMRuntime` can build a TensorRT engine directly from an ONNX file using the system
`nvonnxparser` library:

```swift
import TensorRTLLM

let engine = try TensorRTLLMRuntime().buildEngine(
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
  profiles via ``EngineBuildOptions/profiles``. See <doc:DynamicShapesAndProfiles>.
