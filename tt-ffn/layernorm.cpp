#include <iostream>
#include "layernorm.h"
#define _USE_MATH_DEFINES
#include <math.h>

void LayerNorm(float* input, float* output, const float* scale, const float* bias, int numElements)
{
    // Step 1: Compute the mean
    float sum = 0.0;
    for (int i = 0; i < numElements; i++) {
        sum += input[i];
    }
    float mean = sum / numElements;

    // Step 2: Compute the variance
    float squaredSum = 0.0;
    for (int i = 0; i < numElements; i++) {
        float diff = input[i] - mean;
        squaredSum += diff * diff;
    }
    float variance = squaredSum / numElements;

    // Step 3: Normalize the input
    float invStdDev = 1.0 / std::sqrt(variance + 1e-12);
    for (int i = 0; i < numElements; i++) {
        output[i] = (input[i] - mean) * invStdDev;
    }

    // Step 4: Scale and bias the normalized input
    for (int i = 0; i < numElements; i++) {
        output[i] = output[i] * scale[i] + bias[i];
    }
}

void LayerNorm_derivative(float* input, float* output, const float* scale, const float* bias, int N, int numElements)
{
    // Step 1: Compute the mean
    float sum = 0.0;
    for (int i = 0; i < N*numElements; i++) {
        sum += input[i];
    }
    float mean = sum / N/numElements;

    // Step 2: Compute the variance
    float squaredSum = 0.0;
    for (int i = 0; i < N*numElements; i++) {
        float diff = input[i] - mean;
        squaredSum += diff * diff;
    }
    float variance = squaredSum / N/numElements;

    // Step 3: Compute the derivatives
    float invVar = 1.0 / std::sqrt(variance + 1e-12);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < numElements; j++) {
            // Derivative w.r.t. input
            output[i * numElements + j] = invVar + ((input[i * numElements + j] - mean) * invVar * invVar * invVar / numElements);
            output[i * numElements + j] *= scale[j];
        }
    }
}

