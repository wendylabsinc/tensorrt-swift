# ``TensorRTLLM``

LLM-focused extensions for TensorRT Swift.

`TensorRTLLM` re-exports the core `TensorRT` module and adds streaming helpers intended for
token-by-token generation workflows.

## Overview

Use this module when you want:
- Streaming inference with `AsyncSequence`
- Convenience configuration for autoregressive generation

## Streaming inference

```swift
import TensorRTLLM

let stream = context.stream(
    initialBatch: promptBatch,
    configuration: .init(maxSteps: 100)
) { previousResult in
    // Transform previous output into next input (e.g., append generated token).
    return makeNextBatch(from: previousResult)
}

for try await step in stream {
    print("Step \(step.stepIndex), final: \(step.isFinal)")
    if step.isFinal { break }
}
```

## Topics

### Streaming
- ``ExecutionContext/stream(initialBatch:configuration:nextBatchGenerator:)``
- ``ExecutionContext/streamRepeated(batch:steps:)``
- ``StreamingConfiguration``
- ``StreamingInferenceStep``
- ``InferenceStream``
