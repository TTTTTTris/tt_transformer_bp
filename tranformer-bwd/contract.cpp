#include <iostream>
#include "contract.h"

void clear(
    float* input,
    int size
) {
LOOP_CLEAR:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount max=10*d_model
        input[i] = 0;
    }
}

void add(
    float* input1,
    float* input2,
    int I
) {
LOOP_ADD_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount max=seq_len*d_model * Batchsize
            input1[i] += input2[i];
    }
}

void add_b(
    float* input1,
    float* input2,
    int I,
    int J
) {
LOOP_ADD_B_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount max=seq_len
        LOOP_ADD_B_J :
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount max=d_model * Batchsize
#pragma HLS pipeline off
            input1[i * J + j] += input2[j];
        }
    }
}

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
#pragma HLS loop_tripcount min=1 max=16
        Loop_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount min=8 max=16
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=8 max=16
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=1 max=16
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
#pragma HLS loop_tripcount min=8 max=16
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=1 max=16
                    Loop_R:
                    for (int r = 0; r < R; r++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=10*8 max=16*16
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
#pragma HLS loop_tripcount min=8 max=16
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=1 max=10
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=1 max=10
                    Loop_R:
                    for (int r = 0; r < R; r++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=12 max=12*16
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
            for (int m = 0; m < M; m++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline off
#pragma HLS loop_tripcount min=d_model max=d_hidden
                //float res_l = tensor_l[i * M + m + l_offset]; //IM=IJKL
                //float res_r = tensor_r[m * R + r + r_offset]; //MR=JKLR
                output[i * RANK + r + o_offset] += tensor_l[i * M + m + l_offset] * tensor_r[m * RANK + r + r_offset];
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
    clear(output, RANK * R);
Loop_I:
    for (int i = 0; i < RANK; i++) {
        Loop_R:
        for (int r = 0; r < R; r++) {
#pragma HLS loop_tripcount min=12 max=16
            Loop_M:
            for (int m = 0; m < M; m++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline off
#pragma HLS loop_tripcount min=10*8*8 max=10*12*16
                //float res_l = tensor_l[i * M + m + l_offset]; //MI=JKLI
                //float res_r = tensor_r[m * R + r + r_offset]; //MR=JKLR
                output[i * R + r + o_offset] += tensor_l[m * RANK + i + l_offset] * tensor_r[m * R + r + r_offset];
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
#pragma HLS pipeline off
#pragma HLS loop_tripcount min=8*8*10 max=d_hidden
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
    clear(output, I * J * K * L);

Loop_I:
    for (int i = 0; i < I; i++) {
#pragma HLS loop_tripcount min=10 max=seq_len
        Loop_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount min=8 max=12
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=8 max=16
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=12 max=16
                    Loop_R:
                    for (int r = 0; r < R; r++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline off
#pragma HLS loop_tripcount min=10 max=seq_len
                        //float res_l = tensor_l[i * R + r + l_offset]; //IR
                        //float res_r = tensor_r[r * J * K * L + j * K * L + k * L + l + r_offset]; //RJKL
                        output[i * J * K * L + j * K * L + k * L + l + o_offset] += tensor_l[i * R + r + l_offset] * tensor_r[r * J * K * L + j * K * L + k * L + l + r_offset];
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
#pragma HLS loop_tripcount min=8 max=16
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=8 max=16
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=8 max=12
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
    clear(output, seq_len * J * K * L);

Loop_I:
    for (int i = 0; i < seq_len; i++) {
        Loop_J:
        for (int j = 0; j < J; j++) {
#pragma HLS loop_tripcount min=12 max=16
            Loop_K:
            for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount min=8 max=16
                Loop_L:
                for (int l = 0; l < L; l++) {
#pragma HLS loop_tripcount min=8 max=12
                    Loop_R:
                    for (int r = 0; r < RANK; r++) {
#pragma HLS pipeline off
                        //float res_l = tensor_l[i * R + r + l_offset]; //IR
                        //float res_r = tensor_r[r * J * K * L + j * K * L + k * L + l + r_offset]; //JKLR
                        output[i * J * K * L + j * K * L + k * L + l + o_offset] += tensor_l[i * RANK + r + l_offset] * tensor_r[j * K * L * RANK + k * L * RANK + l * RANK + r + r_offset];
                    }
                }
            }
        }
    }
} // tensor contract to 3-D
