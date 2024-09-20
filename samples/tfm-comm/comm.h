#pragma once
#include <stddef.h>

int acCommInit(const size_t ndims, const size_t* global_nn, const size_t* rr);

int acCommHaloExchange(const size_t ndims, const size_t* nn, const size_t* rr,
                       const size_t nfields);

int acCommQuit(void);
