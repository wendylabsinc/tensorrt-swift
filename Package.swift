// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TensorRTLLM",
    products: [
        .library(
            name: "TensorRTLLM",
            targets: ["TensorRTLLM"]
        ),
        // Beginner Examples
        .executable(name: "HelloTensorRT", targets: ["HelloTensorRT"]),
        .executable(name: "ONNXInference", targets: ["ONNXInference"]),
        .executable(name: "BatchProcessing", targets: ["BatchProcessing"]),
        // Intermediate Examples
        .executable(name: "DynamicBatching", targets: ["DynamicBatching"]),
        .executable(name: "MultiProfile", targets: ["MultiProfile"]),
        .executable(name: "AsyncInference", targets: ["AsyncInference"]),
        .executable(name: "ImageClassifier", targets: ["ImageClassifier"]),
        .executable(name: "DeviceMemoryPipeline", targets: ["DeviceMemoryPipeline"]),
        // Advanced Examples
        .executable(name: "StreamingLLM", targets: ["StreamingLLM"]),
        .executable(name: "MultiGPU", targets: ["MultiGPU"]),
        .executable(name: "CUDAEventPipelining", targets: ["CUDAEventPipelining"]),
        .executable(name: "BenchmarkSuite", targets: ["BenchmarkSuite"]),
        .executable(name: "FP16Quantization", targets: ["FP16Quantization"]),
        // Real-World Examples
        .executable(name: "TextEmbedding", targets: ["TextEmbedding"]),
        .executable(name: "ObjectDetection", targets: ["ObjectDetection"]),
        .executable(name: "WhisperTranscription", targets: ["WhisperTranscription"]),
        .executable(name: "VisionTransformer", targets: ["VisionTransformer"]),
    ],
    targets: [
        .target(
            name: "TensorRTLLMNative",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags(["-std=c++17"], .when(platforms: [.linux])),
                .unsafeFlags(["-I/usr/local/cuda/include"], .when(platforms: [.linux])),
            ],
            linkerSettings: [
                .linkedLibrary("nvinfer", .when(platforms: [.linux])),
                .linkedLibrary("nvinfer_plugin", .when(platforms: [.linux])),
                .linkedLibrary("nvonnxparser", .when(platforms: [.linux])),
                .linkedLibrary("cuda", .when(platforms: [.linux])),
                .linkedLibrary("dl", .when(platforms: [.linux])),
            ]
        ),
        .target(
            name: "TensorRTLLM",
            dependencies: [
                .target(name: "TensorRTLLMNative", condition: .when(platforms: [.linux])),
            ]
        ),
        .testTarget(
            name: "TensorRTLLMTests",
            dependencies: ["TensorRTLLM"]
        ),

        // MARK: - Beginner Examples
        .executableTarget(
            name: "HelloTensorRT",
            dependencies: ["TensorRTLLM"],
            path: "Examples/HelloTensorRT"
        ),
        .executableTarget(
            name: "ONNXInference",
            dependencies: ["TensorRTLLM"],
            path: "Examples/ONNXInference"
        ),
        .executableTarget(
            name: "BatchProcessing",
            dependencies: ["TensorRTLLM"],
            path: "Examples/BatchProcessing"
        ),

        // MARK: - Intermediate Examples
        .executableTarget(
            name: "DynamicBatching",
            dependencies: ["TensorRTLLM"],
            path: "Examples/DynamicBatching"
        ),
        .executableTarget(
            name: "MultiProfile",
            dependencies: ["TensorRTLLM"],
            path: "Examples/MultiProfile"
        ),
        .executableTarget(
            name: "AsyncInference",
            dependencies: ["TensorRTLLM"],
            path: "Examples/AsyncInference"
        ),
        .executableTarget(
            name: "ImageClassifier",
            dependencies: ["TensorRTLLM"],
            path: "Examples/ImageClassifier"
        ),
        .executableTarget(
            name: "DeviceMemoryPipeline",
            dependencies: ["TensorRTLLM"],
            path: "Examples/DeviceMemoryPipeline"
        ),

        // MARK: - Advanced Examples
        .executableTarget(
            name: "StreamingLLM",
            dependencies: ["TensorRTLLM"],
            path: "Examples/StreamingLLM"
        ),
        .executableTarget(
            name: "MultiGPU",
            dependencies: ["TensorRTLLM"],
            path: "Examples/MultiGPU"
        ),
        .executableTarget(
            name: "CUDAEventPipelining",
            dependencies: ["TensorRTLLM"],
            path: "Examples/CUDAEventPipelining"
        ),
        .executableTarget(
            name: "BenchmarkSuite",
            dependencies: ["TensorRTLLM"],
            path: "Examples/BenchmarkSuite"
        ),
        .executableTarget(
            name: "FP16Quantization",
            dependencies: ["TensorRTLLM"],
            path: "Examples/FP16Quantization"
        ),

        // MARK: - Real-World Examples
        .executableTarget(
            name: "TextEmbedding",
            dependencies: ["TensorRTLLM"],
            path: "Examples/TextEmbedding"
        ),
        .executableTarget(
            name: "ObjectDetection",
            dependencies: ["TensorRTLLM"],
            path: "Examples/ObjectDetection"
        ),
        .executableTarget(
            name: "WhisperTranscription",
            dependencies: ["TensorRTLLM"],
            path: "Examples/WhisperTranscription"
        ),
        .executableTarget(
            name: "VisionTransformer",
            dependencies: ["TensorRTLLM"],
            path: "Examples/VisionTransformer"
        ),
    ],
    swiftLanguageModes: [.v6]
)
