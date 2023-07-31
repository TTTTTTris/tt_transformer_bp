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
    clear(output, I * J * K * L);
Loop_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount min=1 max=12
        Loop_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount min=8 max=12
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=8 max=12
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=1 max=12
                    Loop_R:
                    for (int r = 0; r < RANK; r++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline off
                        //float res_l = tensor_l[i * J * R + j * R + r + l_offset]; //IJR
                        //float res_r = tensor_r[r * L * K + k * L + l + r_offset]; //RLK
                        output[i * J * K * L + j * K * L + k * L + l + o_offset] += tensor_l[i * J * RANK + j * RANK + r + l_offset] * tensor_r[r * L * K + k * L + l + r_offset];
                    }
                }
            }
        }
    }
}// tensor contract to 2-D

void contract_left(
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
Loop_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount min=1 max=10
        Loop_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount min=1 max=10
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=8 max=12
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=8 max=12
                    Loop_R:
                    for (int r = 0; r < R; r++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=10*8 max=12*8
                        //float res_l = tensor_l[i * J * R + j * R + r + l_offset]; //RIJ
                        //float res_r = tensor_r[r * L * K + k * L + l + r_offset]; //RKL
                        output[i * J * K * L + j * K * L + k * L + l + o_offset] += tensor_l[r * I * J + i * J + j + l_offset] * tensor_r[r * K * L + k * L + l + r_offset];
                    }
                }
            }
        }
    }
}// tensor contract to 2-D

void contract_right(
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
Loop_I:
    for (int i = 0; i < RANK; i++) {
        Loop_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount min=8 max=12
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=1 max=10
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=1 max=10
                    Loop_R:
                    for (int r = 0; r < R; r++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=12 max=8*12
                        //float res_l = tensor_l[i * J * R + j * R + r + l_offset]; //IJR
                        //float res_r = tensor_r[r * L * K + k * L + l + r_offset]; //KLR
                        output[i * J * K * L + j * K * L + k * L + l + o_offset] += tensor_l[i * J * R + j * R + r + l_offset] * tensor_r[k * L * R + l * R + r + r_offset];
                    }
                }
            }
        }
    }
}// tensor contract to 2-D

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
    clear(output, seq_len * RANK);
Loop_I:
    for (int i = 0; i < seq_len; i++) {
        Loop_R:
        for (int r = 0; r < RANK; r++) {
            Loop_M:
            for (int m = 0; m < d_model; m++) {
#pragma HLS pipeline off
                //float res_l = tensor_l[i * M + m + l_offset]; //IM=IJKL
                //float res_r = tensor_r[m * R + r + r_offset]; //MR=JKLR
                output[i * RANK + r + o_offset] += tensor_l[i * d_model + m + l_offset] * tensor_r[m * RANK + r + r_offset];
            }
        }
    }
} // tensor contract to 1-D

void contract_middle_left(
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
    clear(output, RANK * 12);
Loop_I:
    for (int i = 0; i < RANK; i++) {
        Loop_R:
        for (int r = 0; r < 12; r++) {
            Loop_M:
            for (int m = 0; m < RANK*8*8; m++) {
#pragma HLS loop_flatten off
                //float res_l = tensor_l[i * M + m + l_offset]; //MI=JKLI
                //float res_r = tensor_r[m * R + r + r_offset]; //MR=JKLR
                output[i * 12 + r + o_offset] += tensor_l[m * RANK + i + l_offset] * tensor_r[m * 12 + r + r_offset];
            }
        }
    }
} // tensor contract to 1-D

void contract_middle_right(
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
Loop_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount min=12 max=seq_len 
        Loop_R:
        for (int r = 0; r < RANK; r++) {
            Loop_M:
            for (int m = 0; m < M; m++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=8*8*10 max=d_model
                //float res_l = tensor_l[i * M + m + l_offset]; //IM=IJKL
                //float res_r = tensor_r[m * R + r + r_offset]; //RM=RJKL
                output[i * RANK + r + o_offset] += tensor_l[i * M + m + l_offset] * tensor_r[r * M + m + r_offset];
            }
        }
    }
} // tensor contract to 1-D

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
    clear(output, I * d_model);

Loop_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount min=10 max=seq_len
        Loop_J:
        for (int j = 0; j < 8; j++) {
            Loop_K:
            for (int k = 0; k < 8; k++) {
                Loop_L:
                for (int l = 0; l < 12; l++) {
                    Loop_R:
                    for (int r = 0; r < R; r++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline off
#pragma HLS loop_tripcount min=10 max=seq_len
                        //float res_l = tensor_l[i * R + r + l_offset]; //IR
                        //float res_r = tensor_r[r * J * K * L + j * K * L + k * L + l + r_offset]; //RJKL
                        output[i * d_model + j * 8*12 + k * 12 + l + o_offset] += tensor_l[i * R + r + l_offset] * tensor_r[r * d_model + j *8*12 + k * 12 + l + r_offset];
                    }
                }
            }
        }
    }
} // tensor contract to 3-D

void contract_last_left(
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
    clear(output, RANK * J * K * L);

Loop_I:
    for (int i = 0; i < RANK; i++) {
        Loop_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount min=8 max=12
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=8 max=8
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=8 max=10
                    Loop_R:
                    for (int r = 0; r < R; r++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline off
#pragma HLS loop_tripcount min=12 max=seq_len
                        //float res_l = tensor_l[i * R + r + l_offset]; //RI
                        //float res_r = tensor_r[r * J * K * L + j * K * L + k * L + l + r_offset]; //RJKL
                        output[i * J * K * L + j * K * L + k * L + l + o_offset] += tensor_l[r * RANK + i + l_offset] * tensor_r[r * J * K * L + j * K * L + k * L + l + r_offset];
                    }
                }
            }
        }
    }
} // tensor contract to 3-D

void contract_last_right(
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
    clear(output, seq_len * d_model);

Loop_I:
    for (int i = 0; i < seq_len; i++) {
        Loop_J:
        for (int j = 0; j < 12; j++) {
            Loop_K:
            for (int k = 0; k < 8; k++) {
                Loop_L:
                for (int l = 0; l < 8; l++) {
                    Loop_R:
                    for (int r = 0; r < RANK; r++) {
#pragma HLS pipeline off
                        //float res_l = tensor_l[i * R + r + l_offset]; //IR
                        //float res_r = tensor_r[r * J * K * L + j * K * L + k * L + l + r_offset]; //JKLR
                        output[i * d_model + j * 8*8 + k * 8 + l + o_offset] += tensor_l[i * RANK + r + l_offset] * tensor_r[j * 8*8*RANK + k * 8*RANK + l * RANK + r + r_offset];
                    }
                }
            }
        }
    }
} // tensor contract to 3-D
