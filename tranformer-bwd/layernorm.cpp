#include <iostream>
#include "layernorm.h"
#define _USE_MATH_DEFINES
#include <math.h>

void LayerNorm_derivative(float* input, float* grad_in, float* grad_out, const float* layernorm_w, const float* bias, int N)
{
    float mean[seq_len]{ 0 };
    float var[seq_len]{ 0 };
    float sigma[seq_len]{ 0 };
    Loop_mean_I :
    for (int i = 0; i < N; i++) {
#pragma HLS loop_tripcount max=seq_len
        Loop_mean_J :
        for (int j = 0; j < d_model; j++) {
#pragma HLS LOOP_FLATTEN off
            mean[i] += input[i * d_model + j];
        }
        mean[i] = mean[i] / (d_model);
    }
    // Step 2: Compute the variance
Loop_var_I:
    for (int i = 0; i < N; i++) {
#pragma HLS loop_tripcount max=seq_len
        Loop_var_J :
        for (int j = 0; j < d_model; j++) {
#pragma HLS pipeline off
            var[i] += powf(input[i * d_model + j] - mean[i], 2);
        }
        var[i] = var[i] / (d_model);
    }
Loop_sigma:
    for (int i = 0; i < N; i++) {
#pragma HLS loop_tripcount max=seq_len
        sigma[i] = 1.0 / sqrtf(var[i] + 1e-12);
    }
    // Step 3: Compute the derivatives
    float temp1[seq_len]{ 0 };
    float temp2[seq_len]{ 0 };
Loop_dLN_I1:
    for (int i = 0; i < N; i++) {
#pragma HLS loop_tripcount max=seq_len
        Loop_dLN_J1 :
        for (int j = 0; j < d_model; j++) {
#pragma HLS loop_flatten off
            // Derivative w.r.t. input
            temp1[i] += grad_in[i * d_model + j] * layernorm_w[j];
            temp2[i] += grad_in[i * d_model + j] * (input[i * d_model + j] - mean[i]) * sigma[i] * layernorm_w[j];
        }
    }
    float temp = 0;
Loop_dLN_I2:
    for (int i = 0; i < N; i++) {
#pragma HLS loop_tripcount max=seq_len
        Loop_dLN_J2 :
        for (int j = 0; j < d_model; j++) {
            // Derivative w.r.t. input
            grad_out[i * d_model + j] = grad_in[i * d_model + j] * layernorm_w[j] - (((input[i * d_model + j] - mean[i]) * sigma[i] * temp2[i] + temp1[i]) / (d_model));
            grad_out[i * d_model + j] *= sigma[i];
        }
    }
}

