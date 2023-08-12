#include "order_control.h"
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
#ifdef __SYNTHESIS__
float input[seq_len * 12 * 8 * 8 * Batchsize]{0};
float grad_output[seq_len * 12 * 8 * 8 * Batchsize]{0};
//float grad_cores_ff1[num_cores * WD_FFN]{0};
//float grad_cores_ff2[num_cores * WD_FFN]{0};
//float grad_cores_attnq[num_cores * WD_ATTN]{0};
//float grad_cores_attnk[num_cores * WD_ATTN]{0};
//float grad_cores_attnv[num_cores * WD_ATTN]{0};
//float grad_cores_attnfc[num_cores * WD_ATTN]{0};
int tt_ranks[7];
int attn_shape[6];
int pff_shape_0[6];
int pff_shape_1[6];
#else
extern float input[seq_len * 12 * 8 * 8 * Batchsize];
extern float grad_output[seq_len * 12 * 8 * 8 * Batchsize];
extern float grad_cores_ff1[num_cores * WD_FFN];
extern float grad_cores_ff2[num_cores * WD_FFN];
extern float grad_cores_attnq[num_cores * WD_ATTN];
extern float grad_cores_attnk[num_cores * WD_ATTN];
extern float grad_cores_attnv[num_cores * WD_ATTN];
extern float grad_cores_attnfc[num_cores * WD_ATTN];
extern int tt_ranks[7];
extern int attn_shape[6];
extern int pff_shape_0[6];
extern int pff_shape_1[6];
#endif
/*ATTN paras*/
float grad_output_q[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
float grad_output_k[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
float grad_output_v[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
float grad_output_o[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
float input_q[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_k[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_v[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_o[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_buffer[2][seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float buffer_softmax[4][12 * seq_len * seq_len]{ 0 };
/*FFN paras*/
float input_ff1[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_ff2[seq_len * 16 * 16 * 12 * Batchsize]{ 0 };
float output_1[seq_len * 16 * 16 * 12 * Batchsize]{ 0 };
float output_2[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float grad_output1[seq_len * 12 * 16 * 16 * Batchsize]{ 0 };
float grad_output2[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
/*BUFFER*/
/*compute FC input*/
float buffer_q[2][10 * 8 * 8 * 12]{ 0 };
float buffer_k[2][10 * 8 * 8 * 12]{ 0 };
float buffer_v[2][10 * 8 * 8 * 12]{ 0 };
float buffer_o[2][10 * 8 * 8 * 12]{ 0 };
float buffer[2][10 * 8 * 8 * 12]{ 0 };
/*compute ATTN grad_output*/
float buffer_w2[2][10 * 12 * 16 * 16]{ 0 };
float buffer_w4[2][10 * 12 * 16 * 16]{ 0 };
float buffer_w1[2][10 * 12 * 8 * 8]{ 0 };
float buffer_w3[2][10 * 12 * 8 * 8]{ 0 };

int seq_len_;

void order_control_tt_grad_ff1(
    TYPE_WEIGHT* tt_cores_ff1,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    float grad_cores_1[num_cores * WD_FFN],
    float buffer_w1[2][d_model * 10],
    float buffer_w2[2][d_hidden * 10]
) {
    //compute grad cores for W1
    transpose(buffer_w1[0], buffer_w1[1], seq_len, 10);// 10,32
    contract_last(buffer_w1[1], grad_output, buffer_w2[0], 0, 0, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], pff_shape_0[5], seq_len);//(G1G2G3X)dy 10,12,16,16
    clear(buffer_w1[1], d_model * 10);
    contract_middle_right(grad_output1, buffer_w2[1], buffer_w1[1], 0, 0, 0, seq_len, tt_ranks[4], pff_shape_0[3] * pff_shape_0[4] * pff_shape_0[5]);//(G4G5G6dy) 10
    contract_last_left(buffer_w1[1], input, buffer_w1[0], 0, 0, 0, tt_ranks[4], pff_shape_0[0], pff_shape_0[1], pff_shape_0[2], seq_len);//(G4G5G6dy)X 10,12,8,8
    transpose(buffer_w1[0], buffer_w1[1], tt_ranks[4], pff_shape_0[0] * pff_shape_0[1] * pff_shape_0[2]);//3072, 10
    //gradient_6
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w2[1], WD_FFN * 3, WD_FFN * 4, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], tt_ranks[5], tt_ranks[4]); //G4G5 10,12,16,10
    contract_middle_left(buffer_w2[1], buffer_w2[0], grad_cores_1, 0, 0, WD_FFN * 5, tt_ranks[3], pff_shape_0[5], tt_ranks[5] * pff_shape_0[3] * pff_shape_0[4]);//(G1G2G3X)dyG4G5=gradient_6
    //gradient_4
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w2[1], WD_FFN * 4, WD_FFN * 5, 0, tt_ranks[4], pff_shape_0[4], pff_shape_0[5], tt_ranks[6], tt_ranks[5]); //G5G6 10,12,16,1
    contract_right(buffer_w2[0], buffer_w2[1], grad_cores_1, 0, 0, WD_FFN * 3, tt_ranks[3], pff_shape_0[3], tt_ranks[4], tt_ranks[6], pff_shape_0[4] * pff_shape_0[5]);//(G1G2G3X)dyG5G6=gradient_4
    //gradient_5
    clear(buffer_w2[1], d_hidden * 10);
    contract_left(tt_cores_ff1, buffer_w2[0], buffer_w2[1], WD_FFN * 3, 0, 0, tt_ranks[3], 1, pff_shape_0[4], pff_shape_0[5], tt_ranks[4] * pff_shape_0[3]); //(G1G2G3X)dyG4 10,16,16,1
    contract_right(buffer_w2[1], tt_cores_ff1, grad_cores_1, 0, WD_FFN * 5, WD_FFN * 4, tt_ranks[4], pff_shape_0[4], tt_ranks[5], tt_ranks[6], pff_shape_0[5]);//(G1G2G3X)dyG4G6=gradient_5

    //gradient_3
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w1[0], 0, WD_FFN * 1, 0, tt_ranks[0], pff_shape_0[0], pff_shape_0[1], tt_ranks[2], tt_ranks[1]); //G1G2 1,12,8,10
    contract_left(buffer_w1[0], buffer_w1[1], grad_cores_1, 0, 0, WD_FFN * 2, tt_ranks[0], tt_ranks[2], pff_shape_0[2], tt_ranks[3], pff_shape_0[0] * pff_shape_0[1]);//(G4G5G6dy)XG1G2=gradient_3
    //gradient_1
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w1[0], WD_FFN * 1, WD_FFN * 2, 0, tt_ranks[1], pff_shape_0[1], pff_shape_0[2], tt_ranks[3], tt_ranks[2]); //G2G3 10,8,8,10
    contract_middle_right(buffer_w1[1], buffer_w1[0], grad_cores_1, 0, 0, 0, pff_shape_0[0], tt_ranks[1], tt_ranks[3] * pff_shape_0[1] * pff_shape_0[2]);//(G4G5G6dy)XG2G3=gradient_1
    //gradient_2
    contract_last_left(tt_cores_ff1, buffer_w1[1], buffer_w1[0], 0, 0, 0, tt_ranks[1], pff_shape_0[1], pff_shape_0[2], tt_ranks[2], pff_shape_0[0]);//(G4G5G6dy)XG1 10,8,8,1
    contract_right(buffer_w1[0], tt_cores_ff1, grad_cores_1, 0, WD_FFN * 2, WD_FFN * 1, tt_ranks[1], pff_shape_0[1], tt_ranks[0], tt_ranks[3], tt_ranks[2] * pff_shape_0[2]);//(G4G5G6dy)XG1G3=gradient_2
}

void order_control_tt_grad_ff2(
    TYPE_WEIGHT* tt_cores_ff2,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    float grad_cores_2[num_cores * WD_FFN],
    float buffer_w4[2][d_hidden * 10],
    float buffer_w3[2][d_model * 10]
) {
    //compute grad cores for W2
    transpose(buffer_w4[0], buffer_w4[1], seq_len, 10);// 10,32
    contract_last(buffer_w4[1], grad_output, buffer_w3[0], 0, 0, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], seq_len);//(G1G2G3X)dy 10,8,8,12
    clear(buffer_w4[1], d_hidden * 10);
    contract_middle_right(grad_output2, buffer_w3[1], buffer_w4[1], 0, 0, 0, seq_len, tt_ranks[4], pff_shape_1[3] * pff_shape_1[4] * pff_shape_1[5]);//(G4G5G6dy)
    contract_last_left(buffer_w4[1], input, buffer_w4[0], 0, 0, 0, tt_ranks[4], pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], seq_len);//(G4G5G6dy)X 10,16,16,12
    transpose(buffer_w4[0], buffer_w4[1], tt_ranks[4], pff_shape_1[0] * pff_shape_1[1] * pff_shape_1[2]);//3072, 10
    //gradient_6
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[1], WD_FFN * 3, WD_FFN * 4, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], tt_ranks[5], tt_ranks[4]); //G4G5 10,8,8,10
    contract_middle_left(buffer_w3[1], buffer_w3[0], grad_cores_2, 0, 0, WD_FFN * 5, tt_ranks[3], pff_shape_1[5], tt_ranks[5] * pff_shape_1[3] * pff_shape_1[4]);//(G1G2G3X)dyG4G5=gradient_6
    //gradient_4
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[1], WD_FFN * 4, WD_FFN * 5, 0, tt_ranks[4], pff_shape_1[4], pff_shape_1[5], tt_ranks[6], tt_ranks[5]); //G5G6 10,8,8,1
    contract_right(buffer_w3[0], buffer_w3[1], grad_cores_2, 0, 0, WD_FFN * 3, tt_ranks[3], pff_shape_1[3], tt_ranks[4], tt_ranks[6], pff_shape_1[4] * pff_shape_1[5]);//(G1G2G3X)dyG5G6=gradient_4
    //gradient_5
    clear(buffer_w3[1], d_model * 10);
    contract_left(tt_cores_ff2, buffer_w3[0], buffer_w3[1], WD_FFN * 3, 0, 0, tt_ranks[3], 1, pff_shape_1[4], pff_shape_1[5], tt_ranks[4] * pff_shape_1[3]); //(G1G2G3X)dyG4 10,8,12
    contract_right(buffer_w3[1], tt_cores_ff2, grad_cores_2, 0, WD_FFN * 5, WD_FFN * 4, tt_ranks[4], pff_shape_1[4], tt_ranks[5], tt_ranks[6], pff_shape_1[5]);//(G1G2G3X)dyG4G6=gradient_5

    //gradient_3
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[0], 0, WD_FFN * 1, 0, tt_ranks[0], pff_shape_1[0], pff_shape_1[1], tt_ranks[2], tt_ranks[1]); //G1G2 1,16,16,10
    contract_left(buffer_w4[0], buffer_w4[1], grad_cores_2, 0, 0, WD_FFN * 2, tt_ranks[0], tt_ranks[2], pff_shape_1[2], tt_ranks[3], pff_shape_1[0] * pff_shape_1[1]);//(G4G5G6dy)XG1G2=gradient_3
    //gradient_1
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[0], WD_FFN * 1, WD_FFN * 2, 0, tt_ranks[1], pff_shape_1[1], pff_shape_1[2], tt_ranks[3], tt_ranks[2]); //G2G3
    contract_middle_right(buffer_w4[1], buffer_w4[0], grad_cores_2, 0, 0, 0, pff_shape_1[0], tt_ranks[1], tt_ranks[3] * pff_shape_1[1] * pff_shape_1[2]);//(G4G5G6dy)XG2G3=gradient_1
    //gradient_2
    contract_last_left(tt_cores_ff2, buffer_w4[1], buffer_w4[0], 0, 0, 0, tt_ranks[1], pff_shape_1[1], pff_shape_1[2], tt_ranks[2], pff_shape_1[0]);//(G4G5G6dy)XG1 10,16,16,10
    contract_right(buffer_w4[0], tt_cores_ff2, grad_cores_2, 0, WD_FFN * 2, WD_FFN * 1, tt_ranks[1], pff_shape_1[1], tt_ranks[0], tt_ranks[3], tt_ranks[2] * pff_shape_1[2]);//(G4G5G6dy)XG1G3=gradient_2
}

void order_control_tt_grad_attn(
    /*
    tt_cores: weights represented by TT
    tt_ranks: [r_0,r_1,...,r_d+1]
    tt_shapes: [i_1, i_2,...i_m, j_1,j_2,...,j_n], n + m =d
    input: a vector with shape i_1*i_2*...*i_m
    grad_output: a vector calculated by last layer with shape j_1*j_2*...*j_n
    grad_cores: a arrary stores the result for the gradient to each core
    */
    TYPE_WEIGHT* tt_cores,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    TYPE_WEIGHT* grad_cores,
    float buffer_left[2][10 * 12 * 8 * 8],
    float buffer_right[2][10 * 12 * 8 * 8]
) {
    //compute grad cores for W2
    transpose(buffer_left[0], buffer_right[0], seq_len, 10);// 10,32
    contract_last(buffer_right[0], grad_output, buffer_right[1], 0, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5], seq_len);//(G1G2G3X)dy 10,8,8,12
    clear(buffer_right[0], d_model * 10);
    contract_middle_right(grad_output, buffer_left[1], buffer_right[0], 0, 0, 0, seq_len, tt_ranks[4], attn_shape[3] * attn_shape[4] * attn_shape[5]);//(G4G5G6dy)
    contract_last_left(buffer_right[0], input, buffer_left[0], 0, 0, 0, tt_ranks[4], attn_shape[0], attn_shape[1], attn_shape[2], seq_len);//(G4G5G6dy)X 10,16,16,12
    transpose(buffer_left[0], buffer_right[0], tt_ranks[4], attn_shape[0] * attn_shape[1] * attn_shape[2]);//3072, 10
    //gradient_6
    contract(tt_cores, tt_cores, buffer_left[1], WD_ATTN * 3, WD_ATTN * 4, 0, tt_ranks[3], attn_shape[3], attn_shape[4], tt_ranks[5], tt_ranks[4]); //G4G5 10,8,8,10
    contract_middle_left(buffer_left[1], buffer_right[1], grad_cores, 0, 0, WD_ATTN * 5, tt_ranks[3], attn_shape[5], tt_ranks[5] * attn_shape[3] * attn_shape[4]);//(G1G2G3X)dyG4G5=gradient_6
    //gradient_4
    contract(tt_cores, tt_cores, buffer_left[1], WD_ATTN * 4, WD_ATTN * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6], tt_ranks[5]); //G5G6 10,8,8,1
    contract_right(buffer_right[1], buffer_left[1], grad_cores, 0, 0, WD_ATTN * 3, tt_ranks[3], attn_shape[3], tt_ranks[4], tt_ranks[6], attn_shape[4] * attn_shape[5]);//(G1G2G3X)dyG5G6=gradient_4
    //gradient_5
    clear(buffer_left[1], d_model * 10);
    contract_left(tt_cores, buffer_right[1], buffer_left[1], WD_ATTN * 3, 0, 0, tt_ranks[3], 1, attn_shape[4], attn_shape[5], tt_ranks[4] * attn_shape[3]); //(G1G2G3X)dyG4 10,8,12
    contract_right(buffer_left[1], tt_cores, grad_cores, 0, WD_ATTN * 5, WD_ATTN * 4, tt_ranks[4], attn_shape[4], tt_ranks[5], tt_ranks[6], attn_shape[5]);//(G1G2G3X)dyG4G6=gradient_5

    //gradient_3
    contract(tt_cores, tt_cores, buffer_left[0], 0, WD_ATTN * 1, 0, tt_ranks[0], attn_shape[0], attn_shape[1], tt_ranks[2], tt_ranks[1]); //G1G2 1,16,16,10
    contract_left(buffer_left[0], buffer_right[0], grad_cores, 0, 0, WD_ATTN * 2, tt_ranks[0], tt_ranks[2], attn_shape[2], tt_ranks[3], attn_shape[0] * attn_shape[1]);//(G4G5G6dy)XG1G2=gradient_3
    //gradient_1
    contract(tt_cores, tt_cores, buffer_left[0], WD_ATTN * 1, WD_ATTN * 2, 0, tt_ranks[1], attn_shape[1], attn_shape[2], tt_ranks[3], tt_ranks[2]); //G2G3
    contract_middle_right(buffer_right[0], buffer_left[0], grad_cores, 0, 0, 0, attn_shape[0], tt_ranks[1], tt_ranks[3] * attn_shape[1] * attn_shape[2]);//(G4G5G6dy)XG2G3=gradient_1
    //gradient_2
    contract_last_left(tt_cores, buffer_right[0], buffer_left[0], 0, 0, 0, tt_ranks[1], attn_shape[1], attn_shape[2], tt_ranks[2], attn_shape[0]);//(G4G5G6dy)XG1 10,16,16,10
    contract_right(buffer_left[0], tt_cores, grad_cores, 0, WD_ATTN * 2, WD_ATTN * 1, tt_ranks[1], attn_shape[1], tt_ranks[0], tt_ranks[3], tt_ranks[2] * attn_shape[2]);//(G4G5G6dy)XG1G3=gradient_2
}

void order_control_tt_grad_top(
    TYPE_WEIGHT* tt_cores_attnq,
    TYPE_WEIGHT* tt_cores_attnk,
    TYPE_WEIGHT* tt_cores_attnv,
    TYPE_WEIGHT* tt_cores_attnfc,
    TYPE_WEIGHT* tt_cores_ff1,
    TYPE_WEIGHT* tt_cores_ff2,
    TYPE_WEIGHT* bias_q,
    TYPE_WEIGHT* bias_k,
    TYPE_WEIGHT* bias_v,
    TYPE_WEIGHT* bias_o,
    TYPE_WEIGHT* bias_ff1,
    TYPE_WEIGHT* bias_ff2,
    TYPE_WEIGHT* attn_layernorm_w,
    TYPE_WEIGHT* attn_layernorm_b,
    TYPE_WEIGHT* fc_layernorm_w,
    TYPE_WEIGHT* fc_layernorm_b
) {
///*compute FC input*/
//float buffer_q[2][10 * 8 * 8 * 12]{ 0 };
//float buffer_k[2][10 * 8 * 8 * 12]{ 0 };
//float buffer_v[2][10 * 8 * 8 * 12]{ 0 };
//float buffer_o[2][10 * 8 * 8 * 12]{ 0 };
//float buffer[2][10 * 8 * 8 * 12]{ 0 };
///*compute ATTN grad_output*/
//float buffer_w2[2][10 * 12 * 16 * 16]{ 0 };
//float buffer_w4[2][10 * 12 * 16 * 16]{ 0 };
//float buffer_w1[2][10 * 12 * 8 * 8]{ 0 };
//float buffer_w3[2][10 * 12 * 8 * 8]{ 0 };
#ifdef __SYNTHESIS__
float grad_cores_ff1[num_cores * WD_FFN]{0};
float grad_cores_ff2[num_cores * WD_FFN]{0};
float grad_cores_attnq[num_cores * WD_ATTN]{0};
float grad_cores_attnk[num_cores * WD_ATTN]{0};
float grad_cores_attnv[num_cores * WD_ATTN]{0};
float grad_cores_attnfc[num_cores * WD_ATTN]{0};
#endif
//STORAGE-GRAD-CORES
#pragma HLS BIND_STORAGE variable=grad_cores_attnq type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=grad_cores_attnk type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=grad_cores_attnv type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=grad_cores_attnfc type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=grad_cores_ff1 type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=grad_cores_ff2 type=RAM_1p impl=bram
//STORAGE-ATTN-INOUT
#pragma HLS BIND_STORAGE variable=input type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=grad_output type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=input_q type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=input_k type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=input_v type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=input_o type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=grad_output_q type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=grad_output_k type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=grad_output_v type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=grad_output_o type=RAM_1p impl=bram
//STORAGE-FFN-INOUT
#pragma HLS BIND_STORAGE variable=grad_output1 type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=grad_output2 type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=input_ff1 type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=input_ff2 type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=output_1 type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=output_2 type=RAM_2p impl=bram
//STORAGE-BUFFER
#pragma HLS BIND_STORAGE variable=input_buffer type=RAM_2p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_softmax type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_q type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_k type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_v type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_o type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_w1 type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_w2 type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_w3 type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer_w4 type=RAM_1p impl=bram
#pragma HLS BIND_STORAGE variable=buffer type=RAM_1p impl=bram
//ARRAY_PARTITION
#pragma HLS ARRAY_PARTITION variable=input_buffer type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_softmax type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_q type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_k type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_v type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_o type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_w1 type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_w2 type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_w3 type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_w4 type=complete dim=1
//INTERFACE
#pragma HLS interface mode=bram port=tt_cores_ff1
#pragma HLS interface mode=bram port=tt_cores_ff2
#pragma HLS interface mode=bram port=tt_cores_attnq
#pragma HLS interface mode=bram port=tt_cores_attnk
#pragma HLS interface mode=bram port=tt_cores_attnv
#pragma HLS interface mode=bram port=tt_cores_attnfc
#pragma HLS interface mode=bram port=bias_ff1
#pragma HLS interface mode=bram port=bias_ff2
#pragma HLS interface mode=bram port=bias_q
#pragma HLS interface mode=bram port=bias_k
#pragma HLS interface mode=bram port=bias_v
#pragma HLS interface mode=bram port=bias_o
#pragma HLS interface mode=bram port=attn_layernorm_w
#pragma HLS interface mode=bram port=attn_layernorm_b
#pragma HLS interface mode=bram port=fc_layernorm_w
#pragma HLS interface mode=bram port=fc_layernorm_b

    /*compute ATTN output for FFN input*/
Loop_seq:
    for (int i = 0; i < seq_len; i++) {
        if (abs(input[i * d_model] - 0.0) > 1e-7) {
            seq_len_ = i;
        }
    }
    seq_len_ += 1;
    // compute Q = (G1G2G3G4G5G6)q*X1 + bq
    contract(tt_cores_attnq, tt_cores_attnq, buffer_q[0], 0, WD_ATTN * 1, 0, 1, attn_shape[0], attn_shape[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract(buffer_q[0], tt_cores_attnq, buffer_q[1], 0, WD_ATTN * 2, 0, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3], tt_ranks[2]); //G1G2G3
    contract_middle(input, buffer_q[1], buffer_q[0], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[0] * attn_shape[1] * attn_shape[2]);//G1G2G3X
    contract(tt_cores_attnq, tt_cores_attnq, buffer[0], WD_ATTN * 4, WD_ATTN * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6], tt_ranks[5]); //G5G6
    contract(tt_cores_attnq, buffer[0], buffer_q[1], WD_ATTN * 3, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[4]); //G4G5G6
    contract_last(buffer_q[0], buffer_q[1], input_q, 0, 0, 0, seq_len, attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[3]); //G1G2G3XG4G5G6
    add_b(input_q, bias_q, seq_len_, d_model * Batchsize);//Wqx+bq

    // compute K = (G1G2G3G4G5G6)k*X1 + bk
    contract(tt_cores_attnk, tt_cores_attnk, buffer_k[0], 0, WD_ATTN * 1, 0, 1, attn_shape[0], attn_shape[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract(buffer_k[0], tt_cores_attnk, buffer_k[1], 0, WD_ATTN * 2, 0, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3], tt_ranks[2]); //G1G2G3
    contract_middle(input, buffer_k[1], buffer_k[0], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[0] * attn_shape[1] * attn_shape[2]);//G1G2G3X
    contract(tt_cores_attnk, tt_cores_attnk, buffer[0], WD_ATTN * 4, WD_ATTN * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6], tt_ranks[5]); //G5G6
    contract(tt_cores_attnk, buffer[0], buffer_k[1], WD_ATTN * 3, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[4]); //G4G5G6
    contract_last(buffer_k[0], buffer_k[1], input_k, 0, 0, 0, seq_len, attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[3]); //G1G2G3XG4G5G6
    add_b(input_k, bias_k, seq_len_, d_model * Batchsize);//Wkx+bk

    // compute V = (G1G2G3G4G5G6)v*X1 + bv
    contract(tt_cores_attnv, tt_cores_attnv, buffer_v[0], 0, WD_ATTN * 1, 0, 1, attn_shape[0], attn_shape[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract(buffer_v[0], tt_cores_attnv, buffer_v[1], 0, WD_ATTN * 2, 0, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3], tt_ranks[2]); //G1G2G3
    contract_middle(input, buffer_v[1], buffer_v[0], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[0] * attn_shape[1] * attn_shape[2]);//G1G2G3X
    contract(tt_cores_attnv, tt_cores_attnv, buffer[0], WD_ATTN * 4, WD_ATTN * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6], tt_ranks[5]); //G5G6
    contract(tt_cores_attnv, buffer[0], buffer_v[1], WD_ATTN * 3, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[4]); //G4G5G6
    contract_last(buffer_v[0], buffer_v[1], input_v, 0, 0, 0, seq_len, attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[3]); //G1G2G3XG4G5G6
    add_b(input_v, bias_v, seq_len_, d_model * Batchsize);//Wvx+bv

    // compute xo = softmax(QKT/dk)V
Loop_softmax:
    for (int i = 0; i < seq_len * d_model * Batchsize; i++) {
        input_q[i] /= 8.0;
    }
    transpose_12(input_k, input_buffer[0], seq_len, 12, 64);//(12,32,64)
    transpose_23(input_buffer[0], input_k, 12, seq_len, 64);//(12,64,32)
    transpose_12(input_q, input_buffer[0], seq_len, 12, 64);//(12,32,64)
    MM(input_buffer[0], input_k, 12, buffer_softmax[0], seq_len, seq_len, 64); // QKT/dk //1*12*32*64 1*12*64*32
    softmax(buffer_softmax[0], buffer_softmax[1], 12, seq_len_, seq_len_); //softmax(QKT/dk) 
    transpose_12(input_v, input_buffer[0], seq_len, 12, 64);
    MM(buffer_softmax[1], input_buffer[0], 12, input_buffer[1], seq_len, 64, seq_len); // softmax(QKT/dk)V //1*12*32*32 1*12*32*64

    // compute y2=Woxo+bo
    transpose_12(input_buffer[1], input_buffer[0], 12, seq_len, 64);
    contract(tt_cores_attnfc, tt_cores_attnfc, buffer_o[0], 0, WD_ATTN * 1, 0, 1, attn_shape[0], attn_shape[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract(buffer_o[0], tt_cores_attnfc, buffer[0], 0, WD_ATTN * 2, 0, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3], tt_ranks[2]); //G1G2G3
    contract_middle(input_buffer[0], buffer[0], buffer_o[0], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[0] * attn_shape[1] * attn_shape[2]);//G1G2G3X
    contract(tt_cores_attnfc, tt_cores_attnfc, buffer[1], WD_ATTN * 4, WD_ATTN * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6], tt_ranks[5]);  //G5G6
    contract(tt_cores_attnfc, buffer[1], buffer_o[1], WD_ATTN * 3, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[4]); //G4G5G6
    contract_last(buffer_o[0], buffer_o[1], input_o, 0, 0, 0, seq_len, attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[3]); //G1G2G3XG4G5G6
    add_b(input_o, bias_o, seq_len_, d_model * Batchsize);//Wox+bo
    add(input_o, input, seq_len_ * d_model * Batchsize);//Wox+bo+x

    // compute LN(y2)
    LayerNorm(input_o, input_ff1, attn_layernorm_w, attn_layernorm_b, seq_len_);

    /*compute FFN outputs for ATTN grad_output*/
    // compute x2 = GELU(y1) = GELU(W1x1+b1) = GELU((G1G2G3xG4G5G6)1+b1)
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w1[0], 0, WD_FFN * 1, 0, 1, pff_shape_0[0], pff_shape_0[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract(buffer_w1[0], tt_cores_ff1, buffer_w1[1], 0, WD_FFN * 2, 0, pff_shape_0[0], pff_shape_0[1], pff_shape_0[2], tt_ranks[3], tt_ranks[2]); //G1G2G3
    contract_middle(input_ff1, buffer_w1[1], buffer_w1[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_0[0] * pff_shape_0[1] * pff_shape_0[2]);//G1G2G3X
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w2[0], WD_FFN * 4, WD_FFN * 5, 0, tt_ranks[4], pff_shape_0[4], pff_shape_0[5], tt_ranks[6], tt_ranks[5]); //G5G6
    contract(tt_cores_ff1, buffer_w2[0], buffer_w2[1], WD_FFN * 3, 0, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], pff_shape_0[5], tt_ranks[4]); //G4G5G6
    contract_last(buffer_w1[0], buffer_w2[1], output_1, 0, 0, 0, seq_len, pff_shape_0[3], pff_shape_0[4], pff_shape_0[5], tt_ranks[3]); //G1G2G3XG4G5G6
    add_b(output_1, bias_ff1, seq_len_, d_hidden * Batchsize);//W1x1+b1
    gelu(output_1, input_ff2, seq_len_ * d_hidden * Batchsize);

    // compute y2 = W2x2+b2 = (G1G2G3x2G4G5G6)2 + b2
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[0], 0, WD_FFN * 1, 0, 1, pff_shape_1[0], pff_shape_1[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract(buffer_w4[0], tt_cores_ff2, buffer_w4[1], 0, WD_FFN * 2, 0, pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], tt_ranks[3], tt_ranks[2]); //G1G2G3
    contract_middle(input_ff2, buffer_w4[1], buffer_w4[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_1[0] * pff_shape_1[1] * pff_shape_1[2]);//G1G2G3X
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[0], WD_FFN * 4, WD_FFN * 5, 0, tt_ranks[4], pff_shape_1[4], pff_shape_1[5], tt_ranks[6], tt_ranks[5]);  //G5G6
    contract(tt_cores_ff2, buffer_w3[0], buffer_w3[1], WD_FFN * 3, 0, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], tt_ranks[4]); //G4G5G6
    contract_last(buffer_w4[0], buffer_w3[1], output_2, 0, 0, 0, seq_len, pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], tt_ranks[3]); //G1G2G3XG4G5G6
    add_b(output_2, bias_ff2, seq_len_, d_model * Batchsize);//W1x1+b1
    add(output_2, input_ff1, seq_len_* d_model * Batchsize);//W1x1+b1

    // compute grad_output2 = dy*dLN(y2)/dy2
    LayerNorm_derivative(output_2, grad_output, grad_output2, fc_layernorm_w, fc_layernorm_b, seq_len_);

    // compute grad_output1 = dy*dLN(y2)/dy2*(G4G5G6)2*(G1G2G3)2*dGELU(y1)/dy1
    clear(buffer_w3[0], d_model * 10);
    contract_middle_right(grad_output2, buffer_w3[1], buffer_w3[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_1[3] * pff_shape_1[4] * pff_shape_1[5]);//dy2*dLN(y2)/dy2*(G4G5G6)2 (32,768)*(10,768)
    contract_last_right(buffer_w3[0], buffer_w4[1], grad_output1, 0, 0, 0, seq_len, pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], tt_ranks[3]);//dy2*dLN(y2)/dy2*(G4G5G6)2*(G1G2G3)2 (32,10)*(16,16,12,10)
    gelu_derivative(output_1, grad_output1, seq_len_ * d_hidden * Batchsize);

    // compute grad_output_attn = grad_output1*(G4G5G6)1*(G1G2G3)1
    clear(buffer_w2[0], d_hidden * 10);
    contract_middle_right(grad_output1, buffer_w2[1], buffer_w2[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_0[3] * pff_shape_0[4] * pff_shape_0[5]);//grad_output1*(G4G5G6)1 (32,3072)*(10,3072)
    contract_last_right(buffer_w2[0], buffer_w1[1], grad_output, 0, 0, 0, seq_len, pff_shape_0[0], pff_shape_0[1], pff_shape_0[2], tt_ranks[3]);//grad_output1*(G4G5G6)1*(G1G2G3)1 (32,10)*(12,8,8,10)
    add(grad_output, grad_output2, seq_len_ * d_model * Batchsize);

    /* Train ATTN Layers */
    // compute dy*dLN(y2)/dy2
    LayerNorm_derivative(input_o, grad_output, grad_output_o, attn_layernorm_w, attn_layernorm_b, seq_len_);
     // compute grad_output1 = dy*dLN(yo)/dyo*(G4G5G6)o*(G1G2G3)o
    clear(buffer[1], d_model * 10);
    contract_middle_right(grad_output_o, buffer_o[1], buffer[1], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[3] * attn_shape[4] * attn_shape[5]);//dy2*dLN(y2)/dy2*(G4G5G6)2 (32,768)*(10,768)
    contract_last_right(buffer[1], buffer[0], grad_output, 0, 0, 0, seq_len, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3]);//dy2*dLN(y2)/dy2*(G4G5G6)2*(G1G2G3)2 (32,10)*(12,8,8,10)
    // compute dy*dLN(yo)/dyo*(G4G5G6)o*(G1G2G3)o*V
    transpose_12(grad_output, grad_output_v, seq_len, 12, 64);//(12,32,64)
    transpose_12(input_v, input_buffer[1], seq_len, 12, 64);
    transpose_23(input_buffer[1], input_v, 12, seq_len, 64);
    MM(grad_output_v, input_v, 12, buffer_softmax[0], seq_len, seq_len, 64); //(12,32,64)(12,64,32)

    // compute dy*dLN(yo)/dyo*(G4G5G6)o*(G1G2G3)o*V*d(softmax(QKT))/d(QKT)
    softmax_derivative(buffer_softmax[1], buffer_softmax[0], buffer_softmax[2], buffer_softmax[3], 12, seq_len_, seq_len_);

    // compute dy*dLN(yo)/dyo*(G4G5G6)o*(G1G2G3)o*softmax(QKT)T
    transpose_23(grad_output_v, grad_output, 12, seq_len, 64); //(32,12*64)->(12,64,32)
    MM(grad_output, buffer_softmax[1], 12, grad_output_v, 64, seq_len, seq_len);//(12,64,32)(12,32,32)->(12,64,32)
    transpose_23(grad_output_v, grad_output, 12, 64, seq_len);//(12,32,64)
    transpose_12(grad_output, grad_output_v, 12, seq_len, 64);//(12,32,64)->(32,12,64)

    // compute dy*dLN(yo)/dyo*(G4G5G6)o*(G1G2G3)o*V*d(softmax(QKT))/d(QKT)Q
    transpose_12(input_q, input_buffer[1], seq_len, 12, 64);//(12,32,64)
    transpose_23(input_buffer[1], input_q, 12, seq_len, 64);//(12,64,32)
    MM(input_q, buffer_softmax[3], 12, grad_output_k, 64, seq_len, seq_len); //(12,64,32)(12,32,32)->(12,64,32)
    transpose_23(grad_output_k, grad_output, 12, 64, seq_len);//(12,32,64)
    transpose_12(grad_output, grad_output_k, 12, seq_len, 64);//(12,32,64)->(32,12,64)

    // compute dy*dLN(yo)/dyo*(G4G5G6)o*(G1G2G3)o*V*d(softmax(QKT))/d(QKT)KT
    transpose_23(input_k, input_buffer[1], 12, 64, seq_len);//(12,32,64)
Loop_matmul:
    for (int i = 0; i < seq_len * d_model * Batchsize; i++) {
        input_buffer[1][i] /= 8.0;
    }
    MM(buffer_softmax[3], input_buffer[1], 12, grad_output, seq_len, 64, seq_len); //(12,32,32)(12,32,64)->(12,32,64)
    transpose_12(grad_output, grad_output_q, 12, seq_len, 64);//(12,32,64)->(32,12,64)

    /* COMPUTE GRAD-CORES */
    order_control_tt_grad_ff1(tt_cores_ff1, input_ff1, grad_output1, grad_cores_ff1, buffer_w1, buffer_w2); //[0]G1G2G3X,[1]G1G2G3;[0]G5G6,[1]G4G5G6;
    order_control_tt_grad_ff2(tt_cores_ff2, input_ff2, grad_output2, grad_cores_ff2, buffer_w4, buffer_w3); //[0]G1G2G3X,[1]G1G2G3;[0]xxx,G4G5G6;
    order_control_tt_grad_attn(tt_cores_attnfc, input_buffer[0], grad_output_o, grad_cores_attnfc, buffer_o, buffer); //G1G2G3X; G4G5G6
    order_control_tt_grad_attn(tt_cores_attnq, input, grad_output_q, grad_cores_attnq, buffer_q, buffer);
    order_control_tt_grad_attn(tt_cores_attnk, input, grad_output_k, grad_cores_attnk, buffer_k, buffer);
    order_control_tt_grad_attn(tt_cores_attnv, input, grad_output_v, grad_cores_attnv, buffer_v, buffer);
}
