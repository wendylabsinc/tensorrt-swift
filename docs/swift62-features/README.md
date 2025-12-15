# Swift 6.2+ interop notes for TensorRT Swift bindings

This folder holds the Swift Evolution proposals relevant to TensorRT interop on Jetson (Span, MutableSpan, OutputSpan, Inline Array, UTF8Span).

Key takeaways for this package:
- Use `Span`/`MutableSpan` for zero-copy views onto device/host buffers when bridging to CUDA/TensorRT C++ APIs.
- Prefer `InlineArray` for fixed-size metadata exchanged across the ABI boundary (e.g., strides, dimensions, small vectors) to avoid heap traffic.
- Use `OutputSpan` for APIs that fill caller-provided buffers (e.g., reading engine plan data).
- Adopt UTF8-safe spans when parsing or emitting textual metadata from C++ (e.g., logger messages) to avoid copies.
- Lean on MutableSpan/OutputSpan for refit flows (weights/bias uploads) and calibration cache emission.

The downloaded proposal files mirror Swift 6.2+ behavior so we can wire the C++ interop layer without guessing ABI details.
