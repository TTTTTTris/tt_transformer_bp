#include "contract_last.h"

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