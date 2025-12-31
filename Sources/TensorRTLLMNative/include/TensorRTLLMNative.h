#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, non-zero on failure.
int trt_get_version(int* major, int* minor, int* patch, int* build);

// Plugin registration/loading helpers.
// Many real TensorRT plans require plugin registration before deserialization.
// Returns 0 on success.
int trt_plugins_initialize(void);
// Loads a TensorRT plugin shared library (e.g. custom plugins).
// The library handle is intentionally retained for the lifetime of the process.
// Returns 0 on success.
int trt_plugins_load_library(const char* path);

// CUDA device queries.
// Returns 0 on success.
int trt_cuda_device_count(int32_t* outCount);

// Creates/destroys a TensorRT runtime (nvinfer1::IRuntime).
// Returns 0 on failure.
uintptr_t trt_create_runtime(void);
void trt_destroy_runtime(uintptr_t runtime);

// Creates/destroys a TensorRT builder (nvinfer1::IBuilder).
// Returns 0 on failure.
uintptr_t trt_create_builder(void);
void trt_destroy_builder(uintptr_t builder);

// Builds a trivial FP32 identity engine:
//   input:  float32[elementCount]
//   output: float32[elementCount]
// The returned buffer must be freed with trt_free().
int trt_build_identity_engine_f32(int32_t elementCount, uint8_t** outData, size_t* outSize);

// Builds a trivial FP32 identity engine with a single dynamic dimension:
//   input:  float32[-1]
//   output: float32[-1]
// A single optimization profile is embedded using min/opt/max for the dynamic dimension.
// The returned buffer must be freed with trt_free().
int trt_build_dynamic_identity_engine_f32(int32_t min, int32_t opt, int32_t max, uint8_t** outData, size_t* outSize);

// Builds a trivial FP32 identity engine with a single dynamic dimension and two optimization profiles.
//   input:  float32[-1]
//   output: float32[-1]
// Returns 0 on success.
int trt_build_dual_profile_identity_engine_f32(
  int32_t min0,
  int32_t opt0,
  int32_t max0,
  int32_t min1,
  int32_t opt1,
  int32_t max1,
  uint8_t** outData,
  size_t* outSize
);

// Builds a TensorRT engine from an ONNX file on disk.
// Returns 0 on success and fills a serialized engine plan buffer to be freed with trt_free().
int trt_build_engine_from_onnx_file(
  const char* onnxPath,
  int32_t enableFp16,
  size_t workspaceSizeBytes,
  uint8_t** outData,
  size_t* outSize
);

// Frees buffers returned by TensorRTLLMNative shim (malloc/free).
void trt_free(void* ptr);

// Deserialize a TensorRT engine plan into an engine handle (nvinfer1::ICudaEngine*).
// Returns 0 on failure.
uintptr_t trt_deserialize_engine(const void* data, size_t size);
void trt_destroy_engine(uintptr_t engine);

// Basic IO inspection (TensorRT 8 "bindings" or TensorRT 10 "I/O tensors").
#define TRT_MAX_DIMS 8
#define TRT_MAX_NAME 256
typedef struct trt_io_tensor_desc {
  int32_t isInput;     // 1 = input, 0 = output
  int32_t dataType;    // nvinfer1::DataType numeric value
  int32_t nbDims;      // number of valid dims
  int32_t dims[TRT_MAX_DIMS];
  char name[TRT_MAX_NAME];
} trt_io_tensor_desc;

typedef struct trt_profile_binding_range {
  int32_t profileIndex;
  char const* tensorName;
  int32_t nbDims;
  int32_t minDims[TRT_MAX_DIMS];
  int32_t optDims[TRT_MAX_DIMS];
  int32_t maxDims[TRT_MAX_DIMS];
} trt_profile_binding_range;

// Builds a TensorRT engine from an ONNX file with explicit optimization profiles (dynamic shapes).
// Provide `profileCount > 0` and one or more `profileRanges` entries for each profile/input.
// Returns 0 on success and fills a serialized engine plan buffer to be freed with trt_free().
int trt_build_engine_from_onnx_file_with_profiles(
  const char* onnxPath,
  int32_t enableFp16,
  size_t workspaceSizeBytes,
  const trt_profile_binding_range* profileRanges,
  int32_t profileRangeCount,
  int32_t profileCount,
  uint8_t** outData,
  size_t* outSize
);

int trt_engine_get_io_count(uintptr_t engine, int32_t* outCount);
int trt_engine_get_io_desc(uintptr_t engine, int32_t index, trt_io_tensor_desc* outDesc);
// Returns the number of optimization profiles in the engine.
// Returns 0 on success.
int trt_engine_get_profile_count(uintptr_t engine, int32_t* outCount);

typedef struct trt_named_buffer {
  char const* name;
  void const* data;
  size_t size;
} trt_named_buffer;

typedef struct trt_named_mutable_buffer {
  char const* name;
  void* data;
  size_t size;
} trt_named_mutable_buffer;

// Executes a serialized plan by copying host inputs to device, enqueueing, and copying outputs back to host.
// The caller must provide output buffers of sufficient size.
// Returns 0 on success.
int trt_execute_plan_host(
  const void* plan,
  size_t planSize,
  const trt_named_buffer* inputs,
  int32_t inputCount,
  const trt_named_mutable_buffer* outputs,
  int32_t outputCount
);

// Persistent execution context API (TensorRT 10+).
// Creates a context bound to a serialized plan and a CUDA stream.
// Returns 0 on failure.
uintptr_t trt_context_create(const void* plan, size_t planSize);
// Creates a context bound to a serialized plan on the specified CUDA device (primary context).
uintptr_t trt_context_create_on_device(const void* plan, size_t planSize, int32_t deviceIndex);
// Creates a context bound to a caller-provided CUDA stream (device 0 primary context).
// When `ownsStream` is non-zero, the context will destroy the stream on `trt_context_destroy`.
uintptr_t trt_context_create_with_stream(const void* plan, size_t planSize, uint64_t stream, int32_t ownsStream);
// Same as `trt_context_create_with_stream`, but for the specified CUDA device (primary context).
uintptr_t trt_context_create_with_stream_on_device(const void* plan, size_t planSize, uint64_t stream, int32_t ownsStream, int32_t deviceIndex);
void trt_context_destroy(uintptr_t ctx);

// Returns the CUDA device index used by this context (primary context).
// Returns 0 on success.
int trt_context_get_device_index(uintptr_t ctx, int32_t* outDeviceIndex);

// Sets the active optimization profile on a persistent context (TensorRT 10+).
// Must be called before setting input shapes for that profile.
// Returns 0 on success.
int trt_context_set_optimization_profile(uintptr_t ctx, int32_t profileIndex);

// Sets an input shape on a persistent context (TensorRT 10+).
// Returns 0 on success.
int trt_context_set_input_shape(uintptr_t ctx, const char* inputName, const int32_t* dims, int32_t nbDims);

// Queries the resolved tensor shape on a persistent context (TensorRT 10+).
// Returns 0 on success.
int trt_context_get_tensor_shape(uintptr_t ctx, const char* tensorName, int32_t* outDims, int32_t maxDims, int32_t* outNbDims);

// Executes using a persistent context by copying host inputs to device, enqueueing, and copying outputs back to host.
// Returns 0 on success.
int trt_context_execute_host(
  uintptr_t ctx,
  const trt_named_buffer* inputs,
  int32_t inputCount,
  const trt_named_mutable_buffer* outputs,
  int32_t outputCount
);

// Executes using a persistent context with device-resident buffers.
// Inputs/outputs are interpreted as CUDA device pointers.
// Returns 0 on success.
int trt_context_execute_device(
  uintptr_t ctx,
  const trt_named_buffer* inputs,
  int32_t inputCount,
  const trt_named_mutable_buffer* outputs,
  int32_t outputCount,
  int32_t synchronously
);

// Minimal CUDA driver helpers (for tests and low-level users).
// These use the CUDA primary context for device 0.
int trt_cuda_stream_create(uint64_t* outStream);
int trt_cuda_stream_destroy(uint64_t stream);
int trt_cuda_stream_synchronize(uint64_t stream);

int trt_cuda_event_create(uint64_t* outEvent);
int trt_cuda_event_destroy(uint64_t event);
int trt_cuda_event_record(uint64_t event, uint64_t stream);
int trt_cuda_event_synchronize(uint64_t event);
int trt_cuda_event_query(uint64_t event, int32_t* outReady);

// Records an event on the context's CUDA stream.
int trt_context_record_event(uintptr_t ctx, uint64_t event);

int trt_cuda_malloc(size_t byteCount, uint64_t* outAddress);
int trt_cuda_free(uint64_t address);
int trt_cuda_memcpy_htod(uint64_t dstAddress, const void* src, size_t byteCount);
int trt_cuda_memcpy_dtoh(void* dst, uint64_t srcAddress, size_t byteCount);

// Runs the identity plan on GPU using CUDA driver API.
// Returns 0 on success.
int trt_run_identity_plan_f32(const void* plan, size_t planSize, const float* input, int32_t elementCount, float* output);

#ifdef __cplusplus
} // extern "C"
#endif
