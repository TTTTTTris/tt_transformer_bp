#pragma once
#ifndef SOFTMAX
#define SOFTMAX

#include "defines.h"
#include "clear.h"

void softmax(float* input, float* output, int I, int J, int K);
void softmax_derivative(float* input, float* grad_in, float* grad_inter, float* grad_out, int I, int J, int K);

#endif
#pragma once
