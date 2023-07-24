#pragma once
#ifndef LN
#define LN

#include "defines.h"

void LayerNorm_derivative(float* input, float* grad_in, float* grad_out, const float* layernorm_w, const float* bias, int N);



#endif
#pragma once
