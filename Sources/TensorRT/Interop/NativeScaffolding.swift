import FoundationEssentials

// This file intentionally avoids hard dependencies on TensorRT headers.
// It sketches how Swift 6.2 C++ interop, Span, and InlineArray can be used when wiring to TensorRT.

#if compiler(>=6.2)
/// Placeholder C++ interop helpers for FFI boundaries. Replace these once TensorRT headers are available.
public enum NativeFFI {
    /// Represents a borrowed byte span (host or device). Use `MutableSpan` for outputs.
    public struct ByteSpan {
        public var baseAddress: UnsafeRawPointer?
        public var count: Int

        public init(baseAddress: UnsafeRawPointer?, count: Int) {
            self.baseAddress = baseAddress
            self.count = count
        }
    }

    /// Represents a mutable borrowed byte span.
    public struct MutableByteSpan {
        public var baseAddress: UnsafeMutableRawPointer?
        public var count: Int

        public init(baseAddress: UnsafeMutableRawPointer?, count: Int) {
            self.baseAddress = baseAddress
            self.count = count
        }
    }

    /// Lightweight fixed-size inline array metadata used for shapes/strides.
    public struct InlineIntArray {
        public var storage: [Int]
        public init(_ storage: [Int]) { self.storage = storage }
    }
}
#endif
