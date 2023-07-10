#include <iostream>
#include "GELU.h"
#define _USE_MATH_DEFINES
#include <math.h>

void gelu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = 0.5f * data[i] * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (data[i] + 0.044715f * powf(data[i], 3))));
    }
}

void gelu_derivative(float* data, float*output, int size) {
    for (int i = 0; i < size; i++) {
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (data[i] + 0.044715f * powf(data[i], 3))));
        float pdf = (sqrtf(2.0f / M_PI) * (data[i] + 0.044715f * powf(data[i], 3)) +
            0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (data[i] + 0.044715f * powf(data[i], 3))))) *
            expf(-0.5f * data[i] * data[i]);
        output[i] = cdf + data[i] * pdf;
    }
}