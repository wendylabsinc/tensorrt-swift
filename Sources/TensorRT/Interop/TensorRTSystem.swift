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

    /// Returns the number of CUDA devices visible to the CUDA driver.
    public static func cudaDeviceCount() throws -> Int {
        var count: Int32 = 0
        let status = trt_cuda_device_count(&count)
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to query CUDA device count (status \(status)).")
        }
        return Int(count)
    }

    /// GPU memory information.
    public struct MemoryInfo: Sendable {
        /// Free GPU memory in bytes.
        public var free: Int
        /// Total GPU memory in bytes.
        public var total: Int
        /// Used GPU memory in bytes.
        public var used: Int { total - free }

        /// Free memory as a percentage of total.
        public var freePercentage: Double {
            guard total > 0 else { return 0 }
            return Double(free) / Double(total) * 100
        }

        public init(free: Int, total: Int) {
            self.free = free
            self.total = total
        }
    }

    /// Returns GPU memory information for a specific device.
    ///
    /// Example:
    /// ```swift
    /// let memInfo = try TensorRTSystem.memoryInfo(device: 0)
    /// print("Free: \(memInfo.free / 1_000_000_000) GB")
    /// print("Total: \(memInfo.total / 1_000_000_000) GB")
    /// ```
    public static func memoryInfo(device: Int = 0) throws -> MemoryInfo {
        var free: Int = 0
        var total: Int = 0
        let status = trt_cuda_mem_get_info(Int32(device), &free, &total)
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to query GPU memory info (status \(status)).")
        }
        return MemoryInfo(free: free, total: total)
    }

    /// CUDA device properties.
    public struct DeviceProperties: Sendable {
        /// Device name (e.g., "NVIDIA GeForce RTX 4090").
        public var name: String
        /// Compute capability major version.
        public var computeCapabilityMajor: Int
        /// Compute capability minor version.
        public var computeCapabilityMinor: Int
        /// Compute capability as a string (e.g., "8.9").
        public var computeCapability: String {
            "\(computeCapabilityMajor).\(computeCapabilityMinor)"
        }
        /// Total device memory in bytes.
        public var totalMemory: Int
        /// Number of streaming multiprocessors.
        public var multiProcessorCount: Int
        /// Maximum threads per block.
        public var maxThreadsPerBlock: Int
        /// Warp size (typically 32).
        public var warpSize: Int

        public init(
            name: String,
            computeCapabilityMajor: Int,
            computeCapabilityMinor: Int,
            totalMemory: Int,
            multiProcessorCount: Int,
            maxThreadsPerBlock: Int,
            warpSize: Int
        ) {
            self.name = name
            self.computeCapabilityMajor = computeCapabilityMajor
            self.computeCapabilityMinor = computeCapabilityMinor
            self.totalMemory = totalMemory
            self.multiProcessorCount = multiProcessorCount
            self.maxThreadsPerBlock = maxThreadsPerBlock
            self.warpSize = warpSize
        }
    }

    /// Returns properties for a specific CUDA device.
    ///
    /// Example:
    /// ```swift
    /// let props = try TensorRTSystem.deviceProperties(device: 0)
    /// print("GPU: \(props.name)")
    /// print("Compute Capability: \(props.computeCapability)")
    /// print("Memory: \(props.totalMemory / 1_000_000_000) GB")
    /// ```
    public static func deviceProperties(device: Int = 0) throws -> DeviceProperties {
        var props = trt_device_properties()
        let status = trt_cuda_get_device_properties(Int32(device), &props)
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to query device properties (status \(status)).")
        }

        let name = withUnsafePointer(to: &props.name) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: Int(TRT_DEVICE_NAME_MAX)) { cStr in
                String(cString: cStr)
            }
        }

        return DeviceProperties(
            name: name,
            computeCapabilityMajor: Int(props.computeCapabilityMajor),
            computeCapabilityMinor: Int(props.computeCapabilityMinor),
            totalMemory: props.totalMemory,
            multiProcessorCount: Int(props.multiProcessorCount),
            maxThreadsPerBlock: Int(props.maxThreadsPerBlock),
            warpSize: Int(props.warpSize)
        )
    }

    /// RAII wrapper around a CUDA stream.
    public final class CUDAStream: @unchecked Sendable {
        public let rawValue: UInt64

        /// Creates a new CUDA stream on device 0.
        public init() throws {
            var stream: UInt64 = 0
            let status = trt_cuda_stream_create(&stream)
            guard status == 0, stream != 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to create CUDA stream (status \(status)).")
            }
            self.rawValue = stream
        }

        deinit {
            _ = trt_cuda_stream_destroy(rawValue)
        }

        /// Synchronizes the stream, blocking until all enqueued work completes.
        public func synchronize() throws {
            let status = trt_cuda_stream_synchronize(rawValue)
            guard status == 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to synchronize CUDA stream (status \(status)).")
            }
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
