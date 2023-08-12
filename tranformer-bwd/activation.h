#pragma once
#ifndef GELU
#define GELU

#include "defines.h"

void gelu(float* data, float* output, int size);
void gelu_derivative(float* data, float* output, int size);
void softmax(float* input, float* output, int I, int J, int K);
void softmax_derivative(float* input, float* grad_in, float* grad_inter, float* grad_out, int I, int J, int K);

#endif
#pragma once
