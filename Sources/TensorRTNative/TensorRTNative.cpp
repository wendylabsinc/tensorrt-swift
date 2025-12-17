#include "TensorRTNative.h"

#include <cuda.h>
#include <NvInfer.h>
#include <NvInferVersion.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
class SilentLogger final : public nvinfer1::ILogger {
public:
  void log(Severity, const char*) noexcept override {}
};

SilentLogger& logger() {
  static SilentLogger instance;
  return instance;
}

template <typename T>
void trtDestroy(T* ptr) noexcept {
  if (!ptr) {
    return;
  }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  delete ptr;
#else
  ptr->destroy();
#endif
}

bool copyName(char* dst, size_t dstSize, char const* src) {
  if (!dst || dstSize == 0) {
    return false;
  }
  if (!src) {
    dst[0] = '\0';
    return true;
  }
  std::strncpy(dst, src, dstSize - 1);
  dst[dstSize - 1] = '\0';
  return true;
}

bool fillDims(int32_t* outDims, int32_t maxDims, nvinfer1::Dims const& dims, int32_t* outNbDims) {
  if (!outDims || !outNbDims) {
    return false;
  }
  int32_t nb = dims.nbDims;
  if (nb < 0) {
    return false;
  }
  if (nb > maxDims) {
    nb = maxDims;
  }
  for (int32_t i = 0; i < nb; i++) {
    outDims[i] = dims.d[i];
  }
  for (int32_t i = nb; i < maxDims; i++) {
    outDims[i] = 0;
  }
  *outNbDims = nb;
  return true;
}

CUresult ensureCudaPrimaryContext(CUcontext* outCtx) {
  if (!outCtx) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status = cuInit(0);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  CUdevice device;
  status = cuDeviceGet(&device, 0);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  CUcontext ctx;
  status = cuDevicePrimaryCtxRetain(&ctx, device);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  status = cuCtxSetCurrent(ctx);
  if (status != CUDA_SUCCESS) {
    cuDevicePrimaryCtxRelease(device);
    return status;
  }

  *outCtx = ctx;
  return CUDA_SUCCESS;
}

class CudaPrimaryCtxGuard {
public:
  CudaPrimaryCtxGuard() = default;

  CUresult init() {
    CUresult status = cuInit(0);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    status = cuDeviceGet(&device_, 0);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    status = cuDevicePrimaryCtxRetain(&ctx_, device_);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    retained_ = true;
    return cuCtxSetCurrent(ctx_);
  }

  ~CudaPrimaryCtxGuard() {
    if (retained_) {
      cuDevicePrimaryCtxRelease(device_);
    }
  }

private:
  CUdevice device_{};
  CUcontext ctx_{};
  bool retained_{false};
};

class CudaStreamGuard {
public:
  CudaStreamGuard() = default;

  CUresult init(unsigned int flags = CU_STREAM_DEFAULT) {
    return cuStreamCreate(&stream_, flags);
  }

  ~CudaStreamGuard() {
    if (stream_) {
      cuStreamDestroy(stream_);
    }
  }

  CUstream stream() const { return stream_; }

private:
  CUstream stream_{};
};

struct DeviceBuffer {
  CUdeviceptr ptr{0};
  size_t size{0};
};

struct PersistentExecutionContext {
  CudaPrimaryCtxGuard cuda;
  CudaStreamGuard stream;
  uintptr_t engineHandle{0};
  nvinfer1::ICudaEngine* engine{nullptr};
  nvinfer1::IExecutionContext* exec{nullptr};
  std::unordered_map<std::string, DeviceBuffer> buffers;
};

int ensureDeviceBuffer(PersistentExecutionContext& ctx, const char* name, size_t size, CUdeviceptr* outPtr) {
  if (!name || size == 0 || !outPtr) {
    return 1;
  }
  auto& entry = ctx.buffers[std::string(name)];
  if (entry.ptr != 0 && entry.size == size) {
    *outPtr = entry.ptr;
    return 0;
  }
  if (entry.ptr != 0) {
    cuMemFree(entry.ptr);
    entry.ptr = 0;
    entry.size = 0;
  }
  CUdeviceptr dptr = 0;
  CUresult cu = cuMemAlloc(&dptr, size);
  if (cu != CUDA_SUCCESS) {
    return 2;
  }
  entry.ptr = dptr;
  entry.size = size;
  *outPtr = dptr;
  return 0;
}
} // namespace

int trt_get_version(int* major, int* minor, int* patch, int* build) {
  if (!major || !minor || !patch || !build) {
    return 1;
  }

  // These are exported by libnvinfer as stable C-callable symbols.
  *major = getInferLibMajorVersion();
  *minor = getInferLibMinorVersion();
  *patch = getInferLibPatchVersion();
  *build = getInferLibBuildVersion();
  return (*major > 0) ? 0 : 2;
}

uintptr_t trt_create_runtime(void) {
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger());
  return reinterpret_cast<uintptr_t>(runtime);
}

void trt_destroy_runtime(uintptr_t runtime) {
  if (!runtime) {
    return;
  }
  auto* ptr = reinterpret_cast<nvinfer1::IRuntime*>(runtime);
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  delete ptr;
#else
  ptr->destroy();
#endif
}

uintptr_t trt_create_builder(void) {
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger());
  return reinterpret_cast<uintptr_t>(builder);
}

void trt_destroy_builder(uintptr_t builder) {
  if (!builder) {
    return;
  }
  auto* ptr = reinterpret_cast<nvinfer1::IBuilder*>(builder);
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  delete ptr;
#else
  ptr->destroy();
#endif
}

int trt_build_identity_engine_f32(int32_t elementCount, uint8_t** outData, size_t* outSize) {
  if (!outData || !outSize || elementCount <= 0) {
    return 1;
  }

  *outData = nullptr;
  *outSize = 0;

  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger());
  if (!builder) {
    return 2;
  }

  // Explicit batch is required for modern TensorRT.
  uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);
  if (!network) {
    trtDestroy(builder);
    return 3;
  }

  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = elementCount;
  auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, dims);
  if (!input) {
    trtDestroy(network);
    trtDestroy(builder);
    return 4;
  }

  auto* identity = network->addIdentity(*input);
  if (!identity) {
    trtDestroy(network);
    trtDestroy(builder);
    return 5;
  }

  auto* out = identity->getOutput(0);
  if (!out) {
    trtDestroy(network);
    trtDestroy(builder);
    return 6;
  }
  out->setName("output");
  network->markOutput(*out);

  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  if (!config) {
    trtDestroy(network);
    trtDestroy(builder);
    return 7;
  }

  // Keep build small for tests.
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);

  nvinfer1::IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);
  trtDestroy(config);
  trtDestroy(network);
  trtDestroy(builder);

  if (!serialized || !serialized->data() || serialized->size() == 0) {
    trtDestroy(serialized);
    return 8;
  }

  void* buffer = std::malloc(serialized->size());
  if (!buffer) {
    trtDestroy(serialized);
    return 9;
  }
  std::memcpy(buffer, serialized->data(), serialized->size());
  *outData = reinterpret_cast<uint8_t*>(buffer);
  *outSize = serialized->size();
  trtDestroy(serialized);
  return 0;
}

int trt_build_dynamic_identity_engine_f32(int32_t min, int32_t opt, int32_t max, uint8_t** outData, size_t* outSize) {
  if (!outData || !outSize || min <= 0 || opt <= 0 || max <= 0 || min > opt || opt > max) {
    return 1;
  }

  *outData = nullptr;
  *outSize = 0;

  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger());
  if (!builder) {
    return 2;
  }

  uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);
  if (!network) {
    trtDestroy(builder);
    return 3;
  }

  nvinfer1::Dims inputDims;
  inputDims.nbDims = 1;
  inputDims.d[0] = -1;
  auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, inputDims);
  if (!input) {
    trtDestroy(network);
    trtDestroy(builder);
    return 4;
  }

  auto* identity = network->addIdentity(*input);
  if (!identity) {
    trtDestroy(network);
    trtDestroy(builder);
    return 5;
  }

  auto* out = identity->getOutput(0);
  if (!out) {
    trtDestroy(network);
    trtDestroy(builder);
    return 6;
  }
  out->setName("output");
  network->markOutput(*out);

  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  if (!config) {
    trtDestroy(network);
    trtDestroy(builder);
    return 7;
  }

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);

  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  if (!profile) {
    trtDestroy(config);
    trtDestroy(network);
    trtDestroy(builder);
    return 8;
  }

  nvinfer1::Dims dmin;
  dmin.nbDims = 1;
  dmin.d[0] = min;
  nvinfer1::Dims dopt;
  dopt.nbDims = 1;
  dopt.d[0] = opt;
  nvinfer1::Dims dmax;
  dmax.nbDims = 1;
  dmax.d[0] = max;

  bool ok = true;
  ok = ok && profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, dmin);
  ok = ok && profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, dopt);
  ok = ok && profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, dmax);
  if (!ok) {
    trtDestroy(config);
    trtDestroy(network);
    trtDestroy(builder);
    return 9;
  }

  if (config->addOptimizationProfile(profile) < 0) {
    trtDestroy(config);
    trtDestroy(network);
    trtDestroy(builder);
    return 10;
  }

  nvinfer1::IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);
  trtDestroy(config);
  trtDestroy(network);
  trtDestroy(builder);

  if (!serialized || !serialized->data() || serialized->size() == 0) {
    trtDestroy(serialized);
    return 11;
  }

  void* buffer = std::malloc(serialized->size());
  if (!buffer) {
    trtDestroy(serialized);
    return 12;
  }
  std::memcpy(buffer, serialized->data(), serialized->size());
  *outData = reinterpret_cast<uint8_t*>(buffer);
  *outSize = serialized->size();
  trtDestroy(serialized);
  return 0;
}

void trt_free(void* ptr) {
  std::free(ptr);
}

uintptr_t trt_deserialize_engine(const void* data, size_t size) {
  if (!data || size == 0) {
    return 0;
  }
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger());
  if (!runtime) {
    return 0;
  }

  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(data, size);

  trtDestroy(runtime);

  return reinterpret_cast<uintptr_t>(engine);
}

void trt_destroy_engine(uintptr_t engine) {
  if (!engine) {
    return;
  }
  auto* ptr = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
  trtDestroy(ptr);
}

int trt_engine_get_io_count(uintptr_t engine, int32_t* outCount) {
  if (!engine || !outCount) {
    return 1;
  }
  auto* ptr = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  *outCount = static_cast<int32_t>(ptr->getNbIOTensors());
  return (*outCount >= 0) ? 0 : 2;
#else
  *outCount = static_cast<int32_t>(ptr->getNbBindings());
  return (*outCount >= 0) ? 0 : 2;
#endif
}

int trt_engine_get_io_desc(uintptr_t engine, int32_t index, trt_io_tensor_desc* outDesc) {
  if (!engine || !outDesc || index < 0) {
    return 1;
  }
  std::memset(outDesc, 0, sizeof(*outDesc));

  auto* ptr = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  int32_t count = static_cast<int32_t>(ptr->getNbIOTensors());
  if (index >= count) {
    return 2;
  }

  char const* name = ptr->getIOTensorName(static_cast<int32_t>(index));
  copyName(outDesc->name, TRT_MAX_NAME, name);
  if (!name) {
    return 3;
  }

  outDesc->dataType = static_cast<int32_t>(ptr->getTensorDataType(name));
  nvinfer1::Dims dims = ptr->getTensorShape(name);
  if (!fillDims(outDesc->dims, TRT_MAX_DIMS, dims, &outDesc->nbDims)) {
    return 4;
  }

  auto mode = ptr->getTensorIOMode(name);
  outDesc->isInput = (mode == nvinfer1::TensorIOMode::kINPUT) ? 1 : 0;
  return 0;
#else
  int32_t count = static_cast<int32_t>(ptr->getNbBindings());
  if (index >= count) {
    return 2;
  }

  char const* name = ptr->getBindingName(index);
  copyName(outDesc->name, TRT_MAX_NAME, name);
  outDesc->dataType = static_cast<int32_t>(ptr->getBindingDataType(index));
  nvinfer1::Dims dims = ptr->getBindingDimensions(index);
  if (!fillDims(outDesc->dims, TRT_MAX_DIMS, dims, &outDesc->nbDims)) {
    return 4;
  }
  outDesc->isInput = ptr->bindingIsInput(index) ? 1 : 0;
  return 0;
#endif
}

int trt_execute_plan_host(
  const void* plan,
  size_t planSize,
  const trt_named_buffer* inputs,
  int32_t inputCount,
  const trt_named_mutable_buffer* outputs,
  int32_t outputCount
) {
  if (!plan || planSize == 0 || !inputs || inputCount < 0 || !outputs || outputCount < 0) {
    return 1;
  }

  CudaPrimaryCtxGuard cuda;
  CUresult cu = cuda.init();
  if (cu != CUDA_SUCCESS) {
    return 2;
  }

  CudaStreamGuard stream;
  cu = stream.init();
  if (cu != CUDA_SUCCESS) {
    return 3;
  }

  uintptr_t engineHandle = trt_deserialize_engine(plan, planSize);
  if (!engineHandle) {
    return 4;
  }
  auto* engine = reinterpret_cast<nvinfer1::ICudaEngine*>(engineHandle);

  nvinfer1::IExecutionContext* exec = engine->createExecutionContext();
  if (!exec) {
    trt_destroy_engine(engineHandle);
    return 5;
  }

  // Allocate per-input/output device buffers and set addresses.
  // This is intentionally conservative and assumes the caller provides correct byte sizes.
  // Future work: introspect engine sizes and validate.
  std::vector<CUdeviceptr> allocations;
  allocations.reserve(static_cast<size_t>(inputCount) + static_cast<size_t>(outputCount));

  auto fail = [&](int code) -> int {
    for (CUdeviceptr ptr : allocations) {
      if (ptr) {
        cuMemFree(ptr);
      }
    }
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return code;
  };

  for (int32_t i = 0; i < inputCount; i++) {
    auto const& in = inputs[i];
    if (!in.name || !in.data || in.size == 0) {
      return fail(6);
    }

    CUdeviceptr dptr = 0;
    cu = cuMemAlloc(&dptr, in.size);
    if (cu != CUDA_SUCCESS) {
      return fail(7);
    }
    allocations.push_back(dptr);

    cu = cuMemcpyHtoDAsync(dptr, in.data, in.size, stream.stream());
    if (cu != CUDA_SUCCESS) {
      return fail(8);
    }

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
    if (!exec->setTensorAddress(in.name, reinterpret_cast<void*>(dptr))) {
      return fail(9);
    }
#else
    // Not supported in this minimal shim for TRT < 10.
    return fail(100);
#endif
  }

  for (int32_t i = 0; i < outputCount; i++) {
    auto const& out = outputs[i];
    if (!out.name || !out.data || out.size == 0) {
      return fail(10);
    }

    CUdeviceptr dptr = 0;
    cu = cuMemAlloc(&dptr, out.size);
    if (cu != CUDA_SUCCESS) {
      return fail(11);
    }
    allocations.push_back(dptr);

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
    if (!exec->setTensorAddress(out.name, reinterpret_cast<void*>(dptr))) {
      return fail(12);
    }
#else
    return fail(100);
#endif
  }

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  if (!exec->enqueueV3(reinterpret_cast<cudaStream_t>(stream.stream()))) {
    return fail(13);
  }
#else
  return fail(100);
#endif

  // Copy outputs back to host. Output device allocations are after input allocations.
  int32_t outputBase = inputCount;
  for (int32_t i = 0; i < outputCount; i++) {
    auto const& out = outputs[i];
    CUdeviceptr dptr = allocations[static_cast<size_t>(outputBase + i)];
    cu = cuMemcpyDtoHAsync(out.data, dptr, out.size, stream.stream());
    if (cu != CUDA_SUCCESS) {
      return fail(14);
    }
  }

  cu = cuStreamSynchronize(stream.stream());
  if (cu != CUDA_SUCCESS) {
    return fail(15);
  }

  for (CUdeviceptr ptr : allocations) {
    if (ptr) {
      cuMemFree(ptr);
    }
  }
  trtDestroy(exec);
  trt_destroy_engine(engineHandle);
  return 0;
}

uintptr_t trt_context_create(const void* plan, size_t planSize) {
  if (!plan || planSize == 0) {
    return 0;
  }

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  auto* ctx = new (std::nothrow) PersistentExecutionContext();
  if (!ctx) {
    return 0;
  }

  CUresult cu = ctx->cuda.init();
  if (cu != CUDA_SUCCESS) {
    delete ctx;
    return 0;
  }

  cu = ctx->stream.init();
  if (cu != CUDA_SUCCESS) {
    delete ctx;
    return 0;
  }

  ctx->engineHandle = trt_deserialize_engine(plan, planSize);
  if (!ctx->engineHandle) {
    delete ctx;
    return 0;
  }
  ctx->engine = reinterpret_cast<nvinfer1::ICudaEngine*>(ctx->engineHandle);

  ctx->exec = ctx->engine->createExecutionContext();
  if (!ctx->exec) {
    trt_destroy_engine(ctx->engineHandle);
    ctx->engineHandle = 0;
    delete ctx;
    return 0;
  }

  return reinterpret_cast<uintptr_t>(ctx);
#else
  (void)plan;
  (void)planSize;
  return 0;
#endif
}

void trt_context_destroy(uintptr_t ctxHandle) {
  if (!ctxHandle) {
    return;
  }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  auto* ctx = reinterpret_cast<PersistentExecutionContext*>(ctxHandle);
  for (auto& kv : ctx->buffers) {
    if (kv.second.ptr) {
      cuMemFree(kv.second.ptr);
    }
  }
  ctx->buffers.clear();
  trtDestroy(ctx->exec);
  if (ctx->engineHandle) {
    trt_destroy_engine(ctx->engineHandle);
  }
  delete ctx;
#else
  (void)ctxHandle;
#endif
}

int trt_context_execute_host(
  uintptr_t ctxHandle,
  const trt_named_buffer* inputs,
  int32_t inputCount,
  const trt_named_mutable_buffer* outputs,
  int32_t outputCount
) {
  if (!ctxHandle || !inputs || inputCount < 0 || !outputs || outputCount < 0) {
    return 1;
  }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  auto* ctx = reinterpret_cast<PersistentExecutionContext*>(ctxHandle);
  if (!ctx->exec) {
    return 2;
  }

  for (int32_t i = 0; i < inputCount; i++) {
    auto const& in = inputs[i];
    if (!in.name || !in.data || in.size == 0) {
      return 3;
    }

    CUdeviceptr dptr = 0;
    int rc = ensureDeviceBuffer(*ctx, in.name, in.size, &dptr);
    if (rc != 0) {
      return 4;
    }

    CUresult cu = cuMemcpyHtoDAsync(dptr, in.data, in.size, ctx->stream.stream());
    if (cu != CUDA_SUCCESS) {
      return 5;
    }

    if (!ctx->exec->setTensorAddress(in.name, reinterpret_cast<void*>(dptr))) {
      return 6;
    }
  }

  for (int32_t i = 0; i < outputCount; i++) {
    auto const& out = outputs[i];
    if (!out.name || !out.data || out.size == 0) {
      return 7;
    }

    CUdeviceptr dptr = 0;
    int rc = ensureDeviceBuffer(*ctx, out.name, out.size, &dptr);
    if (rc != 0) {
      return 8;
    }

    if (!ctx->exec->setTensorAddress(out.name, reinterpret_cast<void*>(dptr))) {
      return 9;
    }
  }

  if (!ctx->exec->enqueueV3(reinterpret_cast<cudaStream_t>(ctx->stream.stream()))) {
    return 10;
  }

  for (int32_t i = 0; i < outputCount; i++) {
    auto const& out = outputs[i];
    auto it = ctx->buffers.find(std::string(out.name));
    if (it == ctx->buffers.end() || it->second.ptr == 0) {
      return 11;
    }
    CUresult cu = cuMemcpyDtoHAsync(out.data, it->second.ptr, out.size, ctx->stream.stream());
    if (cu != CUDA_SUCCESS) {
      return 12;
    }
  }

  CUresult cu = cuStreamSynchronize(ctx->stream.stream());
  if (cu != CUDA_SUCCESS) {
    return 13;
  }
  return 0;
#else
  (void)ctxHandle;
  (void)inputs;
  (void)inputCount;
  (void)outputs;
  (void)outputCount;
  return 100;
#endif
}

int trt_context_execute_device(
  uintptr_t ctxHandle,
  const trt_named_buffer* inputs,
  int32_t inputCount,
  const trt_named_mutable_buffer* outputs,
  int32_t outputCount,
  int32_t synchronously
) {
  if (!ctxHandle || !inputs || inputCount < 0 || !outputs || outputCount < 0) {
    return 1;
  }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  auto* ctx = reinterpret_cast<PersistentExecutionContext*>(ctxHandle);
  if (!ctx->exec) {
    return 2;
  }

  for (int32_t i = 0; i < inputCount; i++) {
    auto const& in = inputs[i];
    if (!in.name || !in.data || in.size == 0) {
      return 3;
    }
    if (!ctx->exec->setTensorAddress(in.name, const_cast<void*>(in.data))) {
      return 4;
    }
  }

  for (int32_t i = 0; i < outputCount; i++) {
    auto const& out = outputs[i];
    if (!out.name || !out.data || out.size == 0) {
      return 5;
    }
    if (!ctx->exec->setTensorAddress(out.name, out.data)) {
      return 6;
    }
  }

  if (!ctx->exec->enqueueV3(reinterpret_cast<cudaStream_t>(ctx->stream.stream()))) {
    return 7;
  }

  if (synchronously != 0) {
    CUresult cu = cuStreamSynchronize(ctx->stream.stream());
    if (cu != CUDA_SUCCESS) {
      return 8;
    }
  }
  return 0;
#else
  (void)ctxHandle;
  (void)inputs;
  (void)inputCount;
  (void)outputs;
  (void)outputCount;
  (void)synchronously;
  return 100;
#endif
}

int trt_cuda_malloc(size_t byteCount, uint64_t* outAddress) {
  if (!outAddress || byteCount == 0) {
    return 1;
  }
  CudaPrimaryCtxGuard cuda;
  CUresult cu = cuda.init();
  if (cu != CUDA_SUCCESS) {
    return 2;
  }
  CUdeviceptr ptr = 0;
  cu = cuMemAlloc(&ptr, byteCount);
  if (cu != CUDA_SUCCESS) {
    return 3;
  }
  *outAddress = static_cast<uint64_t>(ptr);
  return 0;
}

int trt_cuda_free(uint64_t address) {
  if (address == 0) {
    return 1;
  }
  CudaPrimaryCtxGuard cuda;
  CUresult cu = cuda.init();
  if (cu != CUDA_SUCCESS) {
    return 2;
  }
  cu = cuMemFree(static_cast<CUdeviceptr>(address));
  return (cu == CUDA_SUCCESS) ? 0 : 3;
}

int trt_cuda_memcpy_htod(uint64_t dstAddress, const void* src, size_t byteCount) {
  if (dstAddress == 0 || !src || byteCount == 0) {
    return 1;
  }
  CudaPrimaryCtxGuard cuda;
  CUresult cu = cuda.init();
  if (cu != CUDA_SUCCESS) {
    return 2;
  }
  cu = cuMemcpyHtoD(static_cast<CUdeviceptr>(dstAddress), src, byteCount);
  return (cu == CUDA_SUCCESS) ? 0 : 3;
}

int trt_cuda_memcpy_dtoh(void* dst, uint64_t srcAddress, size_t byteCount) {
  if (!dst || srcAddress == 0 || byteCount == 0) {
    return 1;
  }
  CudaPrimaryCtxGuard cuda;
  CUresult cu = cuda.init();
  if (cu != CUDA_SUCCESS) {
    return 2;
  }
  cu = cuMemcpyDtoH(dst, static_cast<CUdeviceptr>(srcAddress), byteCount);
  return (cu == CUDA_SUCCESS) ? 0 : 3;
}

int trt_context_set_input_shape(uintptr_t ctxHandle, const char* inputName, const int32_t* dims, int32_t nbDims) {
  if (!ctxHandle || !inputName || !dims || nbDims <= 0) {
    return 1;
  }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  auto* ctx = reinterpret_cast<PersistentExecutionContext*>(ctxHandle);
  if (!ctx->exec) {
    return 2;
  }

  nvinfer1::Dims d;
  d.nbDims = nbDims;
  for (int32_t i = 0; i < nbDims && i < nvinfer1::Dims::MAX_DIMS; i++) {
    d.d[i] = dims[i];
  }
  if (!ctx->exec->setInputShape(inputName, d)) {
    return 3;
  }
  return 0;
#else
  (void)ctxHandle;
  (void)inputName;
  (void)dims;
  (void)nbDims;
  return 100;
#endif
}

int trt_context_get_tensor_shape(uintptr_t ctxHandle, const char* tensorName, int32_t* outDims, int32_t maxDims, int32_t* outNbDims) {
  if (!ctxHandle || !tensorName || !outDims || !outNbDims || maxDims <= 0) {
    return 1;
  }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  auto* ctx = reinterpret_cast<PersistentExecutionContext*>(ctxHandle);
  if (!ctx->exec) {
    return 2;
  }
  nvinfer1::Dims d = ctx->exec->getTensorShape(tensorName);
  if (!fillDims(outDims, maxDims, d, outNbDims)) {
    return 3;
  }
  return 0;
#else
  (void)ctxHandle;
  (void)tensorName;
  (void)outDims;
  (void)maxDims;
  (void)outNbDims;
  return 100;
#endif
}

int trt_run_identity_plan_f32(const void* plan, size_t planSize, const float* input, int32_t elementCount, float* output) {
  if (!plan || planSize == 0 || !input || !output || elementCount <= 0) {
    return 1;
  }

  CudaPrimaryCtxGuard cuda;
  CUresult cu = cuda.init();
  if (cu != CUDA_SUCCESS) {
    return 2;
  }

  CudaStreamGuard stream;
  cu = stream.init();
  if (cu != CUDA_SUCCESS) {
    return 3;
  }

  uintptr_t engineHandle = trt_deserialize_engine(plan, planSize);
  if (!engineHandle) {
    return 4;
  }
  auto* engine = reinterpret_cast<nvinfer1::ICudaEngine*>(engineHandle);

  nvinfer1::IExecutionContext* exec = engine->createExecutionContext();
  if (!exec) {
    trt_destroy_engine(engineHandle);
    return 5;
  }

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = elementCount;
  if (!exec->setInputShape("input", dims)) {
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 6;
  }
#endif

  size_t byteCount = static_cast<size_t>(elementCount) * sizeof(float);
  CUdeviceptr dInput = 0;
  CUdeviceptr dOutput = 0;

  cu = cuMemAlloc(&dInput, byteCount);
  if (cu != CUDA_SUCCESS) {
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 7;
  }
  cu = cuMemAlloc(&dOutput, byteCount);
  if (cu != CUDA_SUCCESS) {
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 8;
  }

  cu = cuMemcpyHtoDAsync(dInput, input, byteCount, stream.stream());
  if (cu != CUDA_SUCCESS) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 9;
  }

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  if (!exec->setTensorAddress("input", reinterpret_cast<void*>(dInput))) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 10;
  }
  if (!exec->setTensorAddress("output", reinterpret_cast<void*>(dOutput))) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 11;
  }

  if (!exec->enqueueV3(reinterpret_cast<cudaStream_t>(stream.stream()))) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 12;
  }
#else
  // Legacy binding API (not used on TRT 10+ in this environment).
  void* bindings[2];
  bindings[0] = reinterpret_cast<void*>(dInput);
  bindings[1] = reinterpret_cast<void*>(dOutput);
  if (!exec->enqueueV2(bindings, reinterpret_cast<cudaStream_t>(stream), nullptr)) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 12;
  }
#endif

  cu = cuMemcpyDtoHAsync(output, dOutput, byteCount, stream.stream());
  if (cu != CUDA_SUCCESS) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 13;
  }

  cu = cuStreamSynchronize(stream.stream());
  if (cu != CUDA_SUCCESS) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    return 14;
  }

  cuMemFree(dOutput);
  cuMemFree(dInput);
  trtDestroy(exec);
  trt_destroy_engine(engineHandle);
  return 0;
}
