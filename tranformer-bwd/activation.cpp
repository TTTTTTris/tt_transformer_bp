#include <iostream>
#include "activation.h"
#define _USE_MATH_DEFINES
#include <math.h>

void gelu(float* data, float*output, int size) {
Loop_gelu:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount max=seq_len
        output[i] = 0.5f * data[i] * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (data[i] + 0.044715f * powf(data[i], 3))));
    }
}

void gelu_derivative(float* data, float*output, int size) {
Loop_dgelu:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount max=seq_len * d_hidden
        float cdf = 1.0f + tanhf(sqrtf(2.0f / M_PI) * (data[i] + 0.044715f * powf(data[i], 3)));
        float pdf = sqrtf(2.0 / M_PI) * expf(-0.5 * data[i] * data[i]);
        output[i] *= 0.5 * (cdf + data[i] * pdf);
    }
}

void softmax(float* input, float* output, int I, int J, int K) {
    float sum[12 * seq_len]{ 0 };
Loop_sum_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount max=12
        Loop_sum_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount max=seq_len
            Loop_sum_K :
            for (int k = 0; k < K; ++k) {
#pragma HLS loop_tripcount max=seq_len
#pragma HLS loop_flatten off
                int index = i * seq_len * seq_len + j * seq_len + k;
                sum[i * seq_len + j] += exp(input[index]);
            }
        }
    }
Loop_sfm_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount max=12
        Loop_sfm_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount max=seq_len
            Loop_sfm_K :
            for (int k = 0; k < K; ++k) {
#pragma HLS loop_tripcount max=seq_len
                int index = i * seq_len * seq_len + j * seq_len + k;
                output[index] = exp(input[index]) / sum[i * seq_len + j];
            }
        }
    }
}

void softmax_derivative(float* input, float* grad_in, float* grad_inter, float* grad_out, int I, int J, int K) {
    //for (int m = 0; m < I * J * K; m++) {
Loop_dsfm_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount max=12
        Loop_dsfm_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount max=seq_len
            Loop_dsfm_K :
            for (int k = 0; k < K; ++k) {
#pragma HLS loop_tripcount max=seq_len
                Loop_dsfm_M :
                for (int m = 0; m < K; ++m) {
#pragma HLS loop_tripcount max=seq_len
#pragma HLS pipeline off
                    int index = i * seq_len * seq_len + j * seq_len + k;
                    int index2 = i * seq_len * seq_len + j * seq_len + m;
                    if (index2 == index) {
                        grad_inter[index] = input[index] * (1 - input[index]);
                    }
                    else {
                        grad_inter[index] = -input[index] * input[index2];
                    }
                    grad_out[index2] += grad_in[index] * grad_inter[index];
                }
            }
        }
    }
}
