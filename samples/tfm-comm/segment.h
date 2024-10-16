#pragma once
#include <stddef.h>

typedef struct {
    size_t ndims;
    size_t* dims;
    size_t* offset;
} Segment;

Segment segment_create(const size_t ndims, const size_t* dims, const size_t* offset);

void segment_destroy(Segment* segment);

void print_segment(const char* label, const Segment segment);

void segment_copy(const Segment in, Segment* out);

void test_segment(void);
