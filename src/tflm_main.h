#pragma once

#include <stddef.h>
#include <stdint.h>

#include "gen/models.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  TFLM_TYPE_UNSUPPORTED,
  TFLM_TYPE_FLOAT32,
  TFLM_TYPE_INT8,
  TFLM_TYPE_UINT8,
} TflmType;

typedef union {
  void *raw;
  float *f32;
  int8_t *i8;
  uint8_t *u8;
} TflmData;

typedef struct {
  TflmType type;
  TflmData data;
  size_t bytes;
  int dimensions;
  const int *shape;
  float scale;
  int32_t zero_point;
} TflmTensor;

typedef enum {
  TFLM_OK,
  TFLM_ERROR,
} TflmStatus;

#define DECLARE_TFLM_INIT(symbol, display_name) \
  TflmStatus tflm_init_##symbol(uint8_t* tensor_arena, size_t tensor_arena_size);
TFLM_FOREACH_MODEL(DECLARE_TFLM_INIT)
#undef DECLARE_TFLM_INIT

TflmTensor* tflm_input(void);
const TflmTensor* tflm_output(void);
size_t tflm_arena_used_bytes(void);
TflmStatus tflm_invoke(void);

#ifdef __cplusplus
}
#endif
