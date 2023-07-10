#include <iostream>
#include "contract.h"

void contract(
    float* tensor_l,
    float* tensor_r,
    float* output,
    int l_offset,
    int r_offset,
    int o_offset,
    int I,
    int J,
    int K,
    int L,
    int R
) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++) {
                for (int l = 0; l < L; l++) {
                    float res = 0;
                    for (int r = 0; r < R; r++) {
                        float res_l = tensor_l[i * J * R + j * R + r + l_offset]; //IJR
                        float res_r = tensor_r[r * L * K + k * L + l + r_offset]; //RLK
                        res +=  res_l * res_r;
                    }
                    output[i * J * K * L + j * K * L + k * L + l + o_offset] = res;
                }
            }
        }
    }
}

void contract_last(
    float* tensor_l,
    float* tensor_r,
    float* output,
    int l_offset,
    int r_offset,
    int o_offset,
    int I,
    int J,
    int K,
    int L,
    int R
) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++) {
                for (int l = 0; l < L; l++) {
                    float res = 0;
                    for (int r = 0; r < R; r++) {
                        float res_l = tensor_l[i * R + r + l_offset]; //IR
                        float res_r = tensor_r[r * J * K * L + j * K * L + k * L + l + r_offset]; //RJKL
                        res +=  res_l * res_r;
                    }
                    output[i * J * K * L + j * K * L + k * L + l + o_offset] = res;
                }
            }
        }
    }
}


void contract_middle(
    float* tensor_l,
    float* tensor_r,
    float* output,
    int l_offset,
    int r_offset,
    int o_offset,
    int I,
    int R,
    int M
) {
    for (int i = 0; i < I; i++) {
        for (int r = 0; r < R; r++) {
                    float res = 0;
                    for (int m = 0; m < M; m++) {
                        float res_l = tensor_l[i * M + m + l_offset]; //IM
                        float res_r = tensor_r[m * R + r + r_offset]; //MR
                        res += res_l * res_r;
                    }
                    output[i * R + r + o_offset] = res;//IR
                }
            }
        }
