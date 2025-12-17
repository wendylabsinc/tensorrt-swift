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

    /// Initializes TensorRT's plugin registry.
    ///
    /// Many real-world TensorRT plans require plugins to be registered before deserialization.
    /// This function is idempotent (safe to call multiple times).
    public static func initializePlugins() throws {
        let status = trt_plugins_initialize()
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("TensorRT plugin initialization failed (status \(status)).")
        }
    }

    /// Loads a shared library that registers TensorRT plugins (e.g. custom layer plugins).
    ///
    /// The library handle is retained for the lifetime of the process.
    public static func loadPluginLibrary(_ path: String) throws {
        let status = path.withCString { cStr in
            trt_plugins_load_library(cStr)
        }
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to load TensorRT plugin library at \(path) (status \(status)).")
        }
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

    /// Builds a small serialized FP32 identity engine plan with a single dynamic dimension.
    ///
    /// This is primarily intended for tests and early prototyping of `reshape(bindings:)`.
    public static func buildDynamicIdentityEnginePlanF32(min: Int, opt: Int, max: Int) throws -> Data {
        var rawPtr: UnsafeMutablePointer<UInt8>?
        var size: Int = 0
        let status = trt_build_dynamic_identity_engine_f32(Int32(min), Int32(opt), Int32(max), &rawPtr, &size)
        guard status == 0, let rawPtr, size > 0 else {
            throw TensorRTError.notImplemented("Failed to build dynamic identity engine plan (status \(status)).")
        }
        defer { trt_free(rawPtr) }
        return Data(bytes: rawPtr, count: size)
    }

    /// Builds a serialized FP32 identity engine plan with two optimization profiles.
    ///
    /// This is primarily intended for tests and early prototyping of runtime profile selection.
    public static func buildDualProfileIdentityEnginePlanF32(
        profile0: (min: Int, opt: Int, max: Int),
        profile1: (min: Int, opt: Int, max: Int)
    ) throws -> Data {
        var rawPtr: UnsafeMutablePointer<UInt8>?
        var size: Int = 0
        let status = trt_build_dual_profile_identity_engine_f32(
            Int32(profile0.min),
            Int32(profile0.opt),
            Int32(profile0.max),
            Int32(profile1.min),
            Int32(profile1.opt),
            Int32(profile1.max),
            &rawPtr,
            &size
        )
        guard status == 0, let rawPtr, size > 0 else {
            throw TensorRTError.notImplemented("Failed to build dual-profile identity engine plan (status \(status)).")
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

    /// RAII wrapper around a CUDA event (CUDA Driver API).
    public final class CUDAEvent: @unchecked Sendable {
        public let rawValue: UInt64

        public init() throws {
            var ev: UInt64 = 0
            let status = trt_cuda_event_create(&ev)
            guard status == 0, ev != 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to create CUDA event (status \(status)).")
            }
            self.rawValue = ev
        }

        deinit {
            _ = trt_cuda_event_destroy(rawValue)
        }

        public func record(on stream: UInt64) throws {
            let status = trt_cuda_event_record(rawValue, stream)
            guard status == 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to record CUDA event (status \(status)).")
            }
        }

        public func synchronize() throws {
            let status = trt_cuda_event_synchronize(rawValue)
            guard status == 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to synchronize CUDA event (status \(status)).")
            }
        }

        public func isReady() throws -> Bool {
            var ready: Int32 = 0
            let status = trt_cuda_event_query(rawValue, &ready)
            guard status == 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to query CUDA event (status \(status)).")
            }
            return ready != 0
        }
    }
}
#endif
