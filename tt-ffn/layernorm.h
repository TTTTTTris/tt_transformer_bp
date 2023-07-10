#pragma once
#ifndef LN
#define LN

#include "defines.h"

void LayerNorm(float* input, float* output, const float* scale, const float* bias, int numElements);
void LayerNorm_derivative(float* input, float* output, const float* scale, const float* bias, int N, int numElements);



#endif
#pragma once
