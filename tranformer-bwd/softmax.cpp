#include <iostream>
#include "softmax.h"
#define _USE_MATH_DEFINES
#include <math.h>

void softmax(float* input, float*output, int I, int J, int K) {
    float sum[12*seq_len]{0};
Loop_sum_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount max=12
        Loop_sum_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount max=seq_len
            Loop_sum_K:
            for (int k = 0; k < K; ++k) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount max=seq_len
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
            Loop_sfm_K:
            for (int k = 0; k < K; ++k) {
#pragma HLS loop_tripcount max=seq_len
                int index = i * seq_len * seq_len + j * seq_len + k;
                output[index] = exp(input[index]) / sum[i*seq_len+j];
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
            Loop_dsfm_K:
            for (int k = 0; k < K; ++k) {
#pragma HLS loop_tripcount max=seq_len
                Loop_dsfm_M:
                for (int m = 0; m < K; ++m) {
#pragma HLS loop_tripcount max=seq_len
#pragma HLS pipeline off
                    int index = i * seq_len * seq_len + j * seq_len + k;
                    int index2 = i * seq_len * seq_len + j * seq_len + m;
                    if (index2 == index) {
                        grad_inter[index] = input[index]*(1-input[index]);
                    }
                    else {
                        grad_inter[index] = -input[index]*input[index2];
                    }
                    grad_out[index2] += grad_in[index] * grad_inter[index];
                }
            }
        }
    }
}
