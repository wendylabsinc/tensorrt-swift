import Glibc

public enum TensorRTRuntimeProbe {
    public struct Version: Sendable, Hashable, CustomStringConvertible {
        public var major: Int
        public var minor: Int
        public var patch: Int
        public var build: Int

        public init(major: Int, minor: Int, patch: Int, build: Int) {
            self.major = major
            self.minor = minor
            self.patch = patch
            self.build = build
        }

        public var description: String {
            "\(major).\(minor).\(patch) (build \(build))"
        }
    }

    /// Attempts to load TensorRT's shared libraries and returns the runtime version.
    ///
    /// This uses `dlopen`/`dlsym` against `libtensorrt_shim.so` (preferred) because it exposes
    /// stable C-callable symbols for querying the TensorRT runtime version.
    public static func inferRuntimeVersion() throws -> Version {
        let candidates = [
            "libtensorrt_shim.so",
            "/lib/x86_64-linux-gnu/libtensorrt_shim.so",
            "libnvinfer.so",
            "/lib/x86_64-linux-gnu/libnvinfer.so",
        ]

        var lastError: String?

        for candidate in candidates {
            guard let handle = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL) else {
                lastError = dlErrorString() ?? "dlopen failed for \(candidate)"
                continue
            }
            defer { dlclose(handle) }

            do {
                let version = try resolveVersion(from: handle)
                return version
            } catch {
                lastError = String(describing: error)
                continue
            }
        }

        throw TensorRTError.runtimeUnavailable(lastError ?? "Unable to locate TensorRT shared libraries")
    }

    private static func resolveVersion(from handle: UnsafeMutableRawPointer) throws -> Version {
        typealias IntFn = @convention(c) () -> CInt

        func load(_ symbol: String) throws -> IntFn {
            guard let sym = dlsym(handle, symbol) else {
                throw TensorRTError.runtimeUnavailable("Missing TensorRT symbol \(symbol): \(dlErrorString() ?? "unknown error")")
            }
            return unsafeBitCast(sym, to: IntFn.self)
        }

        let major = Int(try load("getInferLibMajorVersion")())
        let minor = Int(try load("getInferLibMinorVersion")())
        let patch = Int(try load("getInferLibPatchVersion")())
        let build = Int(try load("getInferLibBuildVersion")())

        if major <= 0 {
            throw TensorRTError.runtimeUnavailable("TensorRT reported invalid major version \(major)")
        }

        return Version(major: major, minor: minor, patch: patch, build: build)
    }

    private static func dlErrorString() -> String? {
        guard let err = dlerror() else { return nil }
        return String(cString: err)
    }
}

