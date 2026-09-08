#include <chrono>
#include <cstdio>
#include <cstring>

#include "src/tflm_main.h"

alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static int RunModel(const char* name) {
#define RUN_IF_MATCH(symbol, display_name)                                   \
  if (std::strcmp(name, display_name) == 0) {                                \
    if (tflm_init_##symbol(tensor_arena, sizeof(tensor_arena)) != TFLM_OK) {  \
      std::fprintf(stderr, "%s: initialization failed\n", name);             \
      return 1;                                                              \
    }                                                                        \
  } else
  TFLM_FOREACH_MODEL(RUN_IF_MATCH) {
    std::fprintf(stderr, "Unknown model: %s\n", name);
    return 1;
  }
#undef RUN_IF_MATCH

  TflmTensor* input = tflm_input();
  const TflmTensor* output = tflm_output();
  std::memset(input->data.raw, 0, input->bytes);

  const auto start = std::chrono::steady_clock::now();
  if (tflm_invoke() != TFLM_OK) {
    std::fprintf(stderr, "%s: invocation failed\n", name);
    return 1;
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start);

  std::printf("%s: input %zu B, output %zu B, arena %zu B, %.3f ms\n",
              name, input->bytes, output->bytes, tflm_arena_used_bytes(),
              elapsed.count());
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s MODEL\n", argv[0]);
    return 1;
  }
  return RunModel(argv[1]);
}
