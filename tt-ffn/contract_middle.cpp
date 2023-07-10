#include "contract_middle.h"

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