#include "TensorRTNative.h"

#include <cuda.h>
#include <NvInfer.h>
#include <NvInferVersion.h>

#include <cstdlib>
#include <cstring>

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

int trt_run_identity_plan_f32(const void* plan, size_t planSize, const float* input, int32_t elementCount, float* output) {
  if (!plan || planSize == 0 || !input || !output || elementCount <= 0) {
    return 1;
  }

  CUcontext ctx;
  CUresult cu = ensureCudaPrimaryContext(&ctx);
  if (cu != CUDA_SUCCESS) {
    return 2;
  }

  CUstream stream;
  cu = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  if (cu != CUDA_SUCCESS) {
    return 3;
  }

  uintptr_t engineHandle = trt_deserialize_engine(plan, planSize);
  if (!engineHandle) {
    cuStreamDestroy(stream);
    return 4;
  }
  auto* engine = reinterpret_cast<nvinfer1::ICudaEngine*>(engineHandle);

  nvinfer1::IExecutionContext* exec = engine->createExecutionContext();
  if (!exec) {
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
    return 5;
  }

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = elementCount;
  if (!exec->setInputShape("input", dims)) {
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
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
    cuStreamDestroy(stream);
    return 7;
  }
  cu = cuMemAlloc(&dOutput, byteCount);
  if (cu != CUDA_SUCCESS) {
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
    return 8;
  }

  cu = cuMemcpyHtoDAsync(dInput, input, byteCount, stream);
  if (cu != CUDA_SUCCESS) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
    return 9;
  }

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  if (!exec->setTensorAddress("input", reinterpret_cast<void*>(dInput))) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
    return 10;
  }
  if (!exec->setTensorAddress("output", reinterpret_cast<void*>(dOutput))) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
    return 11;
  }

  if (!exec->enqueueV3(reinterpret_cast<cudaStream_t>(stream))) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
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
    exec->destroy();
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
    return 12;
  }
#endif

  cu = cuMemcpyDtoHAsync(output, dOutput, byteCount, stream);
  if (cu != CUDA_SUCCESS) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
    return 13;
  }

  cu = cuStreamSynchronize(stream);
  if (cu != CUDA_SUCCESS) {
    cuMemFree(dOutput);
    cuMemFree(dInput);
    trtDestroy(exec);
    trt_destroy_engine(engineHandle);
    cuStreamDestroy(stream);
    return 14;
  }

  cuMemFree(dOutput);
  cuMemFree(dInput);
  trtDestroy(exec);
  trt_destroy_engine(engineHandle);
  cuStreamDestroy(stream);
  return 0;
}
