#pragma once
#ifndef GELU
#define GELU

#include "defines.h"

void gelu(float* data, int size);
void gelu_derivative(float* data, float* output, int size);

#endif
#pragma once
