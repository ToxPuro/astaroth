#pragma once
#include "array.h"

Array backendGetInputTensor(void);

Array backendGetOutputTensor(void);

void backendInit(const size_t domain_length, const size_t radius, const size_t stride);

void backendConvolutionFwd(void);

void backendQuit(void);