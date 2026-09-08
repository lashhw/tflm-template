#include "tflm_main.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace {
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

template <int kMaxOps, typename AddOpsFn>
TflmStatus InitializeModel(const unsigned char* model_data, AddOpsFn add_ops,
                           uint8_t* tensor_arena, size_t tensor_arena_size) {
  if (!tensor_arena || tensor_arena_size == 0)
    return TFLM_ERROR;
  tflite::InitializeTarget();
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION)
    return TFLM_ERROR;

  static tflite::MicroMutableOpResolver<kMaxOps> resolver;
  if (add_ops(resolver) != kTfLiteOk)
    return TFLM_ERROR;
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                               tensor_arena_size);
  if (interpreter.inputs_size() != 1 || interpreter.outputs_size() != 1)
    return TFLM_ERROR;
  if (interpreter.AllocateTensors() != kTfLiteOk)
    return TFLM_ERROR;
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (!input || !output)
    return TFLM_ERROR;
  SetTensor(&g_input, input);
  SetTensor(&g_output, output);
  g_interpreter = &interpreter;
  return TFLM_OK;
}
}  // namespace

TflmTensor* tflm_input(void) { return g_interpreter ? &g_input : nullptr; }
const TflmTensor* tflm_output(void) {
  return g_interpreter ? &g_output : nullptr;
}
size_t tflm_arena_used_bytes(void) {
  return g_interpreter ? g_interpreter->arena_used_bytes() : 0;
}
TflmStatus tflm_invoke(void) {
  if (!g_interpreter)
    return TFLM_ERROR;
  return g_interpreter->Invoke() == kTfLiteOk ? TFLM_OK : TFLM_ERROR;
}

#define DEFINE_TFLM_INIT(symbol, display_name)                         \
  TflmStatus tflm_init_##symbol(uint8_t* tensor_arena,                 \
                                size_t tensor_arena_size) {            \
    using Resolver =                                                   \
        tflite::MicroMutableOpResolver<TFLM_MODEL_OP_COUNT_##symbol>;  \
    auto add_ops = [](Resolver& resolver) {                            \
      TFLM_APPLY_MODEL_OPS_##symbol(resolver);                         \
      return kTfLiteOk;                                                \
    };                                                                 \
    static const TflmStatus status =                                   \
        InitializeModel<TFLM_MODEL_OP_COUNT_##symbol>(                 \
            g_model_data_##symbol, add_ops, tensor_arena,              \
            tensor_arena_size);                                        \
    return status;                                                     \
  }
TFLM_FOREACH_MODEL(DEFINE_TFLM_INIT)
#undef DEFINE_TFLM_INIT
