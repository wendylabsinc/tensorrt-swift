// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TensorRT",
    products: [
        .library(
            name: "TensorRT",
            targets: ["TensorRT"]
        ),
    ],
    targets: [
        .target(
            name: "TensorRTNative",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags(["-std=c++17"], .when(platforms: [.linux])),
                .unsafeFlags(["-I/usr/local/cuda/include"], .when(platforms: [.linux])),
            ],
            linkerSettings: [
                .linkedLibrary("nvinfer", .when(platforms: [.linux])),
                .linkedLibrary("cuda", .when(platforms: [.linux])),
                .linkedLibrary("dl", .when(platforms: [.linux])),
            ]
        ),
        .target(
            name: "TensorRT",
            dependencies: [
                .target(name: "TensorRTNative", condition: .when(platforms: [.linux])),
            ]
        ),
        .testTarget(
            name: "TensorRTTests",
            dependencies: ["TensorRT"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
