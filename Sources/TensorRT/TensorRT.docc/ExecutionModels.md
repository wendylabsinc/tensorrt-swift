# Execution Models

This package supports multiple execution styles depending on how you want to manage memory and CUDA
streams.

## Host buffers

Use ``ExecutionContext/enqueue(_:synchronously:)`` with `.host(Data)` inputs:

```swift
let result = try await ctx.enqueue(batch, synchronously: true)
```

This path copies host inputs to device, enqueues, and copies outputs back to host.

## Device pointers

Use ``ExecutionContext/enqueueDevice(inputs:outputs:synchronously:)`` when you already manage device
buffers (CUDA device pointers):

```swift
try await ctx.enqueueDevice(
    inputs: ["input": (address: dIn, length: byteCount)],
    outputs: ["output": (address: dOut, length: byteCount)],
    synchronously: false
)
```

## External streams and CUDA events

When using an external CUDA stream via ``ExecutionQueue/external(streamIdentifier:)``:

- Pass `synchronously: false` and control completion yourself.
- Record a CUDA event via ``ExecutionContext/recordEvent(_:)`` and wait on it.

```swift
let ctx = try engine.makeExecutionContext(queue: .external(streamIdentifier: stream))
try await ctx.enqueueDevice(..., synchronously: false)

let event = try TensorRTSystem.CUDAEvent()
try await ctx.recordEvent(event)
try event.synchronize()
```

## Device selection

To target a non-zero GPU, set ``DeviceSelection/gpu`` at build/load time:

```swift
let engine = try TensorRTRuntime().deserializeEngine(
    from: plan,
    configuration: EngineLoadConfiguration(device: DeviceSelection(gpu: 1))
)
```
