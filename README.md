# TFLM Template

This template embeds `.tflite` models, builds TensorFlow Lite Micro static
libraries, and provides a host smoke-test runner.

## Set up

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Add models and build

Place one or more models in `src/models/`, then build the library for the
target you need:

| Target | Output |
| --- | --- |
| `make microlite` | `gen/linux_x86_64_debug_gcc/lib/libtensorflow-microlite.a` |
| `make microlite-f7` | `gen/f7/lib/libtensorflow-microlite.a` with Cortex-M7 CMSIS-NN kernels |
| `make microlite-h7` | `gen/h7/lib/libtensorflow-microlite.a` with Cortex-M7 CMSIS-NN kernels |
| `make` | All libraries |

The model filename stem becomes its API suffix. For example,
`hello_world_int8.tflite` generates `tflm_init_hello_world_int8()`.

## Use the library from C

Add `src/` to the application's include paths and link the target library.

```c
#include "tflm_main.h"

_Alignas(16) static uint8_t arena[256 * 1024];

if (tflm_init_hello_world_int8(arena, sizeof(arena)) != TFLM_OK)
  return ERROR;

TflmTensor *input = tflm_input();
const TflmTensor *output = tflm_output();
int8_t *input_data = input->data.i8;

/* Fill input_data, then run inference. */
if (tflm_invoke() != TFLM_OK)
  return ERROR;
const int8_t *output_data = output->data.i8;
```

The arena must remain valid for the application's lifetime. Only one model is
active at a time, and initialization must complete before using the tensor
pointers. `tflm_arena_used_bytes()` reports the number of arena bytes used.

## Check inference on the host

Build and run the host runner:

```sh
make tflm_main
gen/tflm_main
```

The runner fills each input tensor with zero bytes and invokes every generated
model once through the same API shown above. It reports tensor sizes, arena use,
and latency.

## Clean

Run `make clean` to remove generated model sources and build outputs.
