#pragma once
#include <stddef.h>
#include <stdint.h>

#define NTUPLE_MAX_NELEMS ((size_t)4)

typedef struct {
    size_t nelems;
    uint64_t elems[NTUPLE_MAX_NELEMS];
} Ntuple;

Ntuple make_ntuple(const size_t nelems);
Ntuple make_ntuple_with_elems(const size_t nelems, const uint64_t* elems);

void ntuple_fill(const uint64_t value, Ntuple* ntuple);
Ntuple ntuple_add(const Ntuple a, const Ntuple b);
Ntuple ntuple_sub(const Ntuple a, const Ntuple b);
Ntuple ntuple_mul(const uint64_t a, const Ntuple b);

void print_ntuple(const char* label, const Ntuple ntuple);
int test_ntuple(void);
