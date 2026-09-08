#include <new>

#include "tflm_main.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace {
alignas(tflite::MicroInterpreter) unsigned char g_interpreter_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* g_interpreter = nullptr;
TflmTensor g_input = {};
TflmTensor g_output = {};

TflmType ConvertType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32: return TFLM_TYPE_FLOAT32;
    case kTfLiteInt8: return TFLM_TYPE_INT8;
    case kTfLiteUInt8: return TFLM_TYPE_UINT8;
    default: return TFLM_TYPE_UNSUPPORTED;
  }
}

void SetTensor(TflmTensor* target, TfLiteTensor* source) {
  target->type = ConvertType(source->type);
  target->data.raw = source->data.raw;
  target->bytes = source->bytes;
  target->dimensions = source->dims->size;
  target->shape = source->dims->data;
  target->scale = source->params.scale;
  target->zero_point = source->params.zero_point;
}

#define TFLM_ENSURE(condition) \
  do {                         \
    if (!(condition))          \
      return TFLM_ERROR;       \
  } while (false)

template <int kMaxOps, typename AddOpsFn>
TflmStatus InitializeModel(const unsigned char* model_data, AddOpsFn add_ops,
                           uint8_t* tensor_arena, size_t tensor_arena_size) {
  TFLM_ENSURE(tensor_arena && tensor_arena_size > 0);
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(model_data);
  TFLM_ENSURE(model->version() == TFLITE_SCHEMA_VERSION);

  static tflite::MicroMutableOpResolver<kMaxOps> resolver;
  static const TfLiteStatus resolver_status = add_ops(resolver);
  TFLM_ENSURE(resolver_status == kTfLiteOk);

  if (g_interpreter) {
    g_interpreter->~MicroInterpreter();
    g_interpreter = nullptr;
  }
  g_input = {};
  g_output = {};

  g_interpreter = new (g_interpreter_storage) tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size);
  TFLM_ENSURE(g_interpreter->initialization_status() == kTfLiteOk);
  TFLM_ENSURE(g_interpreter->inputs_size() == 1 && g_interpreter->outputs_size() == 1);
  TFLM_ENSURE(g_interpreter->AllocateTensors() == kTfLiteOk);

  TfLiteTensor* input = g_interpreter->input(0);
  TfLiteTensor* output = g_interpreter->output(0);
  TFLM_ENSURE(input && output);

  SetTensor(&g_input, input);
  SetTensor(&g_output, output);

  return TFLM_OK;
}
}  // namespace

#define DEFINE_TFLM_INIT(symbol, display_name)                                      \
  TflmStatus tflm_init_##symbol(uint8_t* tensor_arena, size_t tensor_arena_size) {  \
    using Resolver = tflite::MicroMutableOpResolver<TFLM_MODEL_OP_COUNT_##symbol>;  \
    auto add_ops = [](Resolver& resolver) {                                         \
      TFLM_APPLY_MODEL_OPS_##symbol(resolver);                                      \
      return kTfLiteOk;                                                             \
    };                                                                              \
    return InitializeModel<TFLM_MODEL_OP_COUNT_##symbol>(                           \
        g_model_data_##symbol, add_ops, tensor_arena, tensor_arena_size);           \
  }
TFLM_FOREACH_MODEL(DEFINE_TFLM_INIT)
#undef DEFINE_TFLM_INIT

TflmTensor* tflm_input(void) {
  return g_interpreter ? &g_input : nullptr;
}

const TflmTensor* tflm_output(void) {
  return g_interpreter ? &g_output : nullptr;
}

size_t tflm_arena_used_bytes(void) {
  return g_interpreter ? g_interpreter->arena_used_bytes() : 0;
}

TflmStatus tflm_invoke(void) {
  return g_interpreter && g_interpreter->Invoke() == kTfLiteOk ? TFLM_OK : TFLM_ERROR;
}
