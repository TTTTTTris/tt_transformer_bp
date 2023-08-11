#pragma once
#ifndef TRANSPOSE
#define TRANSPOSE

#include "defines.h"

void transpose(float* input, float* output, int I, int J);
void transpose_12(float* input, float* output, int I, int J, int K);
void transpose_23(float* input, float* output, int I, int J, int K);

#endif
