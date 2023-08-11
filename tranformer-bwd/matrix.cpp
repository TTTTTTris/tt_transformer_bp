#include <iostream>
#include "matrix.h"

using namespace std;

void MM(float* matrix1, float* matrix2, int M, float* result, int I, int J, int K)
{
    clear(result, M * I * J);
Loop_M:
    for (int m = 0; m < M; ++m){
#pragma HLS loop_tripcount max=12
    Loop_I:
        for (int i = 0; i < I; ++i){
#pragma HLS loop_tripcount min=seq_len max=64
        Loop_J:
            for (int j = 0; j < J; ++j){
#pragma HLS loop_tripcount min=seq_len max=64
            Loop_K:
                for (int k = 0; k < K; ++k){
#pragma HLS loop_tripcount min=seq_len max=64
#pragma HLS pipeline off
                    //float value1 = matrix1[m * I * K + i * K + k]; //MIK
                    //float value2 = matrix2[m * J * K + k * J + j]; //MKJ
                    result[m * I * J + i * J + j] += matrix1[m * I * K + i * K + k] * matrix2[m * J * K + k * J + j];
                }
            }
        }
    }
}

