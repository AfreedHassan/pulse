// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "WhisperHelper",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "pulse-whisper-coreml", targets: ["WhisperHelper"]),
    ],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.9.0"),
    ],
    targets: [
        .executableTarget(
            name: "WhisperHelper",
            dependencies: ["WhisperKit"],
            path: "Sources"
        ),
    ]
)
