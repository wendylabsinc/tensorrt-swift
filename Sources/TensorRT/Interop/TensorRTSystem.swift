import FoundationEssentials

#if canImport(TensorRTNative)
import TensorRTNative

/// Direct bindings to the system-installed TensorRT library (via a small C++ shim).
///
/// This is intentionally minimal: it exists to prove linkage, validate runtime availability,
/// and provide basic lifecycle primitives for future engine/context support.
public enum TensorRTSystem {
    /// Returns the TensorRT runtime version reported by the linked `libnvinfer.so`.
    public static func linkedRuntimeVersion() throws -> TensorRTRuntimeProbe.Version {
        var major: Int32 = 0
        var minor: Int32 = 0
        var patch: Int32 = 0
        var build: Int32 = 0

        let status = trt_get_version(&major, &minor, &patch, &build)
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("Unable to query TensorRT version from linked libnvinfer (status \(status)).")
        }

        return TensorRTRuntimeProbe.Version(
            major: Int(major),
            minor: Int(minor),
            patch: Int(patch),
            build: Int(build)
        )
    }

    /// Builds a small serialized FP32 engine plan for a trivial identity network.
    public static func buildIdentityEnginePlan(elementCount: Int = 8) throws -> Data {
        var rawPtr: UnsafeMutablePointer<UInt8>?
        var size: Int = 0
        let status = trt_build_identity_engine_f32(Int32(elementCount), &rawPtr, &size)
        guard status == 0, let rawPtr, size > 0 else {
            throw TensorRTError.notImplemented("Failed to build identity engine plan (status \(status)).")
        }
        defer { trt_free(rawPtr) }
        return Data(bytes: rawPtr, count: size)
    }

    /// Builds a small serialized FP32 engine plan for a trivial identity network.
    ///
    /// - Note: Once `OutputSpan`-based container initializers are available in the toolchain,
    ///   this can be updated to initialize the return buffer without using an "unsafe uninitialized"
    ///   closure.
    public static func buildIdentityEnginePlanBytes(elementCount: Int = 8) throws -> [UInt8] {
        var rawPtr: UnsafeMutablePointer<UInt8>?
        var size: Int = 0
        let status = trt_build_identity_engine_f32(Int32(elementCount), &rawPtr, &size)
        guard status == 0, let rawPtr, size > 0 else {
            throw TensorRTError.notImplemented("Failed to build identity engine plan (status \(status)).")
        }
        defer { trt_free(rawPtr) }

        return Array(unsafeUninitializedCapacity: size) { buffer, count in
            if let base = buffer.baseAddress {
                base.initialize(from: rawPtr, count: size)
            }
            count = size
        }
    }

    /// Executes an identity plan on the GPU and returns the output.
    public static func runIdentityPlanF32(plan: Data, input: [Float]) throws -> [Float] {
        guard !input.isEmpty else { return [] }
        var output = Array<Float>(repeating: 0, count: input.count)

        let status: Int32 = plan.bytes.withUnsafeBytes { planBytes in
            input.span.withUnsafeBufferPointer { inputPtr in
                var outputSpan = output.mutableSpan
                return outputSpan.withUnsafeMutableBufferPointer { outputPtr in
                    trt_run_identity_plan_f32(
                        planBytes.baseAddress,
                        planBytes.count,
                        inputPtr.baseAddress,
                        Int32(input.count),
                        outputPtr.baseAddress
                    )
                }
            }
        }

        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("TensorRT identity execution failed (status \(status)).")
        }
        return output
    }

    /// RAII wrapper around `nvinfer1::IRuntime`.
    public final class Runtime: @unchecked Sendable {
        fileprivate let handle: UInt

        public init() throws {
            let handle = trt_create_runtime()
            guard handle != 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to create TensorRT runtime (createInferRuntime returned null).")
            }
            self.handle = handle
        }

        deinit {
            trt_destroy_runtime(handle)
        }
    }

    /// RAII wrapper around `nvinfer1::IBuilder`.
    public final class Builder: @unchecked Sendable {
        fileprivate let handle: UInt

        public init() throws {
            let handle = trt_create_builder()
            guard handle != 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to create TensorRT builder (createInferBuilder returned null).")
            }
            self.handle = handle
        }

        deinit {
            trt_destroy_builder(handle)
        }
    }
}
#endif
