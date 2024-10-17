#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t ndims;
    uint64_t* dims;
    uint64_t* offset;
} Segment;

Segment segment_create(const size_t ndims, const uint64_t* dims, const uint64_t* offset);

void segment_destroy(Segment* segment);

void print_segment(const char* label, const Segment segment);

Segment segment_dup(const Segment in);

void test_segment(void);

#define PRINTD_SEGMENT(segment) (print_segment(#segment, (segment)))
