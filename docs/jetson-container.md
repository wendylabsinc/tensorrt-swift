# Running in a Jetson Container (Orin Nano, AGX Thor)

This project supports Jetson devices when the container matches the host JetPack release and provides
CUDA and TensorRT libraries for aarch64.

## 1) Check JetPack and L4T release

On the Jetson host:

```bash
head -n1 /etc/nv_tegra_release
```

Pick a container tag that matches the host L4T release (for example, r36.x for JetPack 6).

## 2) Base image options (aarch64)

Recommended base images:

- `nvcr.io/nvidia/l4t-jetpack:<l4t-tag>` (includes CUDA, TensorRT, and dev tooling)
- `nvcr.io/nvidia/l4t-tensorrt:<l4t-tag>` (smaller, focused on TensorRT)

Choose the tag that matches your host L4T release. Mismatched tags often lead to driver or library errors.

## 3) Example Dockerfile (Jetson)

```Dockerfile
FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0

RUN apt-get update && apt-get install -y \
    curl \
    libxml2 \
    libcurl4 \
    libedit2 \
    libsqlite3-0 \
    libc6-dev \
    binutils \
    libgcc-11-dev \
    libstdc++-11-dev \
    zlib1g-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Swift 6.2+
RUN curl -L https://swiftlang.github.io/swiftly/swiftly-install.sh | bash
RUN /root/.swiftly/bin/swiftly install 6.2
ENV PATH="/root/.swiftly/bin:/root/.swiftly/toolchains/swift-6.2/usr/bin:${PATH}"

WORKDIR /workspace
```

If you need a different JetPack/L4T release, update the base image tag accordingly.

## 4) Run the container

From the host (Jetson):

```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc=host \
  -v "$PWD":/workspace \
  -w /workspace \
  nvcr.io/nvidia/l4t-jetpack:r36.2.0 \
  bash
```

Inside the container:

```bash
./scripts/swiftw test
./scripts/swiftw run HelloTensorRT
```

## 5) Common issues

- `libnvinfer.so` not found: ensure the container tag matches JetPack, and that the TensorRT packages are present.
- `CUDA driver version is insufficient`: the container tag is newer than the host driver/JetPack.
- Swift not found: check `PATH` or install a Swift aarch64 toolchain with Swiftly.

If you need to pin a custom build output path:

```bash
SWIFT_BUILD_PATH=/workspace/.build ./scripts/swiftw test
```
