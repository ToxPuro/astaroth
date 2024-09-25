#include "tgarray.h"

// Type-specific definitions
#define TG_OUTPUT_SOURCE
#define TG_DTYPE int
#include "tgarray_prototype.h"
#undef TG_DTYPE

#define TG_DTYPE double
#include "tgarray_prototype.h"
#undef TG_DTYPE
#undef TG_OUTPUT_SOURCE
