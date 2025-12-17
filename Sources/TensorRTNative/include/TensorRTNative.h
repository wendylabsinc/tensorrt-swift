#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, non-zero on failure.
int trt_get_version(int* major, int* minor, int* patch, int* build);

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

// Frees buffers returned by TensorRTNative shim (malloc/free).
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

int trt_engine_get_io_count(uintptr_t engine, int32_t* outCount);
int trt_engine_get_io_desc(uintptr_t engine, int32_t index, trt_io_tensor_desc* outDesc);

// Runs the identity plan on GPU using CUDA driver API.
// Returns 0 on success.
int trt_run_identity_plan_f32(const void* plan, size_t planSize, const float* input, int32_t elementCount, float* output);

#ifdef __cplusplus
} // extern "C"
#endif
