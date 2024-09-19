#pragma once
#include <stddef.h>

#include "buffer.h"

typedef size_t FIELDDATATYPE;

typedef struct {
    size_t ndims;
    size_t* dims;
    size_t* offset;

    size_t nfields;

    Buffer buffer;
} PackedData;

PackedData acCreatePackedData(const size_t ndims, const size_t* dims, const size_t* offset,
                              const size_t nfields);

void acDestroyPackedData(PackedData* data);
