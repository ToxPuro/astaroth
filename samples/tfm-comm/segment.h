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

// typedef struct {
//     size_t ndims;
//     size_t* min;
//     size_t* max;
// } BoundingBox;

// BoundingBox bounding_box_create(const size_t ndims, const size_t* min, const size_t* max);

// void bounding_box_destroy(BoundingBox* bounding_box);

// void print_bounding_box(const char* label, const BoundingBox bounding_box);

// void test_bounding_box(void);

// BoundingBox bounding_box_create_from_segment(const Segment segment);

// Segment segment_create_from_bounding_box(const BoundingBox bounding_box);
