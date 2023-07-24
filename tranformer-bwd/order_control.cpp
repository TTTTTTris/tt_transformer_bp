#include "order_control.h"
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;

//extern float input[seq_len * 12 * 8 * 8 * Batchsize];
//extern float grad_output[seq_len * 12 * 8 * 8 * Batchsize];
float input[seq_len * 12 * 8 * 8 * Batchsize];
float grad_output[seq_len * 12 * 8 * 8 * Batchsize];
float grad_output_q[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
float grad_output_k[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
float grad_output_v[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
float grad_output_o[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
float input_q[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_k[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_v[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_o[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float input_buffer[2][seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float buffer_softmax[4][12 * seq_len * seq_len]{0};
//float buffer_score[2][seq_len * 8 * 8 * 12 * Batchsize]{0};
int seq_len_;

void order_control_tt_grad(
/*
tt_cores: weights represented by TT
tt_ranks: [r_0,r_1,...,r_d+1]
tt_shapes: [i_1, i_2,...i_m, j_1,j_2,...,j_n], n + m =d
input: a vector with shape i_1*i_2*...*i_m
grad_output: a vector calculated by last layer with shape j_1*j_2*...*j_n
grad_cores: a arrary stores the result for the gradient to each core
*/
    TYPE_WEIGHT* tt_cores,
    int tt_ranks[7],
    int tt_shapes[6],
    TYPE_DATA* grad_output,
    TYPE_WEIGHT* grad_cores,
    float buffer_left[2][10 * 12 * 8 * 8],
    float buffer_right[2][10 * 12 * 8 * 8]
) {
    //compute grad cores for W2
    transpose(buffer_left[0], buffer_right[0], seq_len, 10);// 10,32
    contract_last(buffer_right[0], grad_output, buffer_right[1], 0, 0, 0, tt_ranks[3], tt_shapes[3], tt_shapes[4], tt_shapes[5], seq_len);//(G1G2G3X)dy 10,8,8,12
    clear(buffer_right[0], d_model * 10);
    contract_middle_right(grad_output, buffer_left[1], buffer_right[0], 0, 0, 0, seq_len, tt_ranks[4], tt_shapes[3] * tt_shapes[4] * tt_shapes[5]);//(G4G5G6dy)
    contract_last_left(buffer_right[0], input, buffer_left[0], 0, 0, 0, tt_ranks[4], tt_shapes[0], tt_shapes[1], tt_shapes[2], seq_len);//(G4G5G6dy)X 10,16,16,12
    transpose(buffer_left[0], buffer_right[0], tt_ranks[4], tt_shapes[0] * tt_shapes[1] * tt_shapes[2]);//3072, 10
    //gradient_6
    contract(tt_cores, tt_cores, buffer_left[1], WD * 3, WD * 4, 0, tt_ranks[3], tt_shapes[3], tt_shapes[4], tt_ranks[5]); //G4G5 10,8,8,10, tt_ranks[4]
    contract_middle_left(buffer_left[1], buffer_right[1], grad_cores, 0, 0, WD * 5, tt_ranks[3], tt_shapes[5], tt_ranks[5] * tt_shapes[3] * tt_shapes[4]);//(G1G2G3X)dyG4G5=gradient_6
    //gradient_4
    contract(tt_cores, tt_cores, buffer_left[1], WD * 4, WD * 5, 0, tt_ranks[4], tt_shapes[4], tt_shapes[5], tt_ranks[6]); //G5G6 10,8,8,1, tt_ranks[5]
    contract_right(buffer_right[1], buffer_left[1], grad_cores, 0, 0, WD * 3, tt_ranks[3], tt_shapes[3], tt_ranks[4], tt_ranks[6], tt_shapes[4] * tt_shapes[5]);//(G1G2G3X)dyG5G6=gradient_4
    //gradient_5
    clear(buffer_left[1], d_model * 10);
    contract_left(tt_cores, buffer_right[1], buffer_left[1], WD * 3, 0, 0, tt_ranks[3], 1, tt_shapes[4], tt_shapes[5], tt_ranks[4] * tt_shapes[3]); //(G1G2G3X)dyG4 10,8,12
    contract_right(buffer_left[1], tt_cores, grad_cores, 0, WD * 5, WD * 4, tt_ranks[4], tt_shapes[4], tt_ranks[5], tt_ranks[6], tt_shapes[5]);//(G1G2G3X)dyG4G6=gradient_5

    //gradient_3
    contract(tt_cores, tt_cores, buffer_left[0], 0, WD * 1, 0, tt_ranks[0], tt_shapes[0], tt_shapes[1], tt_ranks[2]); //G1G2 1,16,16,10, tt_ranks[1]
    contract_left(buffer_left[0], buffer_right[0], grad_cores, 0, 0, WD * 2, tt_ranks[0], tt_ranks[2], tt_shapes[2], tt_ranks[3], tt_shapes[0] * tt_shapes[1]);//(G4G5G6dy)XG1G2=gradient_3
    //gradient_1
    contract(tt_cores, tt_cores, buffer_left[0], WD * 1, WD * 2, 0, tt_ranks[1], tt_shapes[1], tt_shapes[2], tt_ranks[3]); //G2G3, tt_ranks[2]
    contract_middle_right(buffer_right[0], buffer_left[0], grad_cores, 0, 0, 0, tt_shapes[0], tt_ranks[1], tt_ranks[3] * tt_shapes[1] * tt_shapes[2]);//(G4G5G6dy)XG2G3=gradient_1
    //gradient_2
    contract_last_left(tt_cores, buffer_right[0], buffer_left[0], 0, 0, 0, tt_ranks[1], tt_shapes[1], tt_shapes[2], tt_ranks[2], tt_shapes[0]);//(G4G5G6dy)XG1 10,16,16,10
    contract_right(buffer_left[0], tt_cores, grad_cores, 0, WD * 2, WD * 1, tt_ranks[1], tt_shapes[1], tt_ranks[0], tt_ranks[3], tt_ranks[2] * tt_shapes[2]);//(G4G5G6dy)XG1G3=gradient_2
}

void order_control_tt_grad_o(
    /*
    tt_cores: weights represented by TT
    tt_ranks: [r_0,r_1,...,r_d+1]
    tt_shapes: [i_1, i_2,...i_m, j_1,j_2,...,j_n], n + m =d
    input: a vector with shape i_1*i_2*...*i_m
    grad_output: a vector calculated by last layer with shape j_1*j_2*...*j_n
    grad_cores: a arrary stores the result for the gradient to each core
    */
    TYPE_WEIGHT* tt_cores,
    int tt_ranks[7],
    int tt_shapes[6],
    TYPE_DATA* grad_output,
    TYPE_WEIGHT* grad_cores,
    float buffer_left[2][10 * 12 * 8 * 8],
    float buffer_right[2][10 * 12 * 8 * 8]
) {
    //compute grad cores for W2
    transpose(buffer_left[0], buffer_right[0], seq_len, 10);// 10,32
    contract_last(buffer_right[0], grad_output, buffer_right[1], 0, 0, 0, tt_ranks[3], tt_shapes[3], tt_shapes[4], tt_shapes[5], seq_len);//(G1G2G3X)dy 10,8,8,12
    clear(buffer_right[0], d_model * 10);
    contract_middle_right(grad_output, buffer_left[1], buffer_right[0], 0, 0, 0, seq_len, tt_ranks[4], tt_shapes[3] * tt_shapes[4] * tt_shapes[5]);//(G4G5G6dy)
    contract_last_left(buffer_right[0], input_buffer[0], buffer_left[0], 0, 0, 0, tt_ranks[4], tt_shapes[0], tt_shapes[1], tt_shapes[2], seq_len);//(G4G5G6dy)X 10,16,16,12
    transpose(buffer_left[0], buffer_right[0], tt_ranks[4], tt_shapes[0] * tt_shapes[1] * tt_shapes[2]);//3072, 10
    //gradient_6
    contract(tt_cores, tt_cores, buffer_left[1], WD * 3, WD * 4, 0, tt_ranks[3], tt_shapes[3], tt_shapes[4], tt_ranks[5]); //G4G5 10,8,8,10, tt_ranks[4]
    contract_middle_left(buffer_left[1], buffer_right[1], grad_cores, 0, 0, WD * 5, tt_ranks[3], tt_shapes[5], tt_ranks[5] * tt_shapes[3] * tt_shapes[4]);//(G1G2G3X)dyG4G5=gradient_6
    //gradient_4
    contract(tt_cores, tt_cores, buffer_left[1], WD * 4, WD * 5, 0, tt_ranks[4], tt_shapes[4], tt_shapes[5], tt_ranks[6]); //G5G6 10,8,8,1, tt_ranks[5]
    contract_right(buffer_right[1], buffer_left[1], grad_cores, 0, 0, WD * 3, tt_ranks[3], tt_shapes[3], tt_ranks[4], tt_ranks[6], tt_shapes[4] * tt_shapes[5]);//(G1G2G3X)dyG5G6=gradient_4
    //gradient_5
    clear(buffer_left[1], d_model * 10);
    contract_left(tt_cores, buffer_right[1], buffer_left[1], WD * 3, 0, 0, tt_ranks[3], 1, tt_shapes[4], tt_shapes[5], tt_ranks[4] * tt_shapes[3]); //(G1G2G3X)dyG4 10,8,12
    contract_right(buffer_left[1], tt_cores, grad_cores, 0, WD * 5, WD * 4, tt_ranks[4], tt_shapes[4], tt_ranks[5], tt_ranks[6], tt_shapes[5]);//(G1G2G3X)dyG4G6=gradient_5

    //gradient_3
    contract(tt_cores, tt_cores, buffer_left[0], 0, WD * 1, 0, tt_ranks[0], tt_shapes[0], tt_shapes[1], tt_ranks[2]); //G1G2 1,16,16,10, tt_ranks[1]
    contract_left(buffer_left[0], buffer_right[0], grad_cores, 0, 0, WD * 2, tt_ranks[0], tt_ranks[2], tt_shapes[2], tt_ranks[3], tt_shapes[0] * tt_shapes[1]);//(G4G5G6dy)XG1G2=gradient_3
    //gradient_1
    contract(tt_cores, tt_cores, buffer_left[0], WD * 1, WD * 2, 0, tt_ranks[1], tt_shapes[1], tt_shapes[2], tt_ranks[3]); //G2G3, tt_ranks[2]
    contract_middle_right(buffer_right[0], buffer_left[0], grad_cores, 0, 0, 0, tt_shapes[0], tt_ranks[1], tt_ranks[3] * tt_shapes[1] * tt_shapes[2]);//(G4G5G6dy)XG2G3=gradient_1
    //gradient_2
    contract_last_left(tt_cores, buffer_right[0], buffer_left[0], 0, 0, 0, tt_ranks[1], tt_shapes[1], tt_shapes[2], tt_ranks[2], tt_shapes[0]);//(G4G5G6dy)XG1 10,16,16,10
    contract_right(buffer_left[0], tt_cores, grad_cores, 0, WD * 2, WD * 1, tt_ranks[1], tt_shapes[1], tt_ranks[0], tt_ranks[3], tt_ranks[2] * tt_shapes[2]);//(G4G5G6dy)XG1G3=gradient_2
}

void order_control_tt_grad_attn(
    TYPE_WEIGHT* tt_cores_attnq,
    TYPE_WEIGHT* tt_cores_attnk,
    TYPE_WEIGHT* tt_cores_attnv,
    TYPE_WEIGHT* tt_cores_attnfc,
    TYPE_WEIGHT* bias_q,
    TYPE_WEIGHT* bias_k,
    TYPE_WEIGHT* bias_v,
    TYPE_WEIGHT* bias_o,
    TYPE_WEIGHT* layernorm_w,
    TYPE_WEIGHT* layernorm_b,
    int tt_ranks[7],
    int attn_shape[6],
    TYPE_WEIGHT* grad_cores_attnq,
    TYPE_WEIGHT* grad_cores_attnk,
    TYPE_WEIGHT* grad_cores_attnv,
    TYPE_WEIGHT* grad_cores_attnfc
) {
    float buffer_q[2][10 * 8 * 8 * 12]{ 0 };
    float buffer_k[2][10 * 8 * 8 * 12]{ 0 };
    float buffer_v[2][10 * 8 * 8 * 12]{ 0 };
    float buffer_o[2][10 * 8 * 8 * 12]{ 0 };
    float buffer[2][10 * 8 * 8 * 12]{ 0 };
//INPUT
#pragma HLS BIND_STORAGE variable=input_buffer impl=bram type=RAM_2P
#pragma HLS ARRAY_PARTITION variable=input_buffer type=complete dim=1
#pragma HLS BIND_STORAGE variable=buffer_softmax impl=bram type=RAM_2P
#pragma HLS ARRAY_PARTITION variable=buffer_softmax type=complete dim=1
#pragma HLS BIND_STORAGE variable=input impl=bram type=RAM_2P
#pragma HLS BIND_STORAGE variable=input_q impl=bram type=RAM_2P
#pragma HLS BIND_STORAGE variable=input_k impl=bram type=RAM_2P
#pragma HLS BIND_STORAGE variable=input_v impl=bram type=RAM_2P
#pragma HLS BIND_STORAGE variable=input_o impl=bram type=RAM_2P
//GRADOUT
#pragma HLS BIND_STORAGE variable=grad_output impl=bram type=RAM_2P
#pragma HLS BIND_STORAGE variable=grad_output_q impl=bram type=RAM_2P
#pragma HLS BIND_STORAGE variable=grad_output_k impl=bram type=RAM_2P
#pragma HLS BIND_STORAGE variable=grad_output_v impl=bram type=RAM_2P
#pragma HLS BIND_STORAGE variable=grad_output_o impl=bram type=RAM_2P
//BUFFER
#pragma HLS BIND_STORAGE variable=buffer_q impl=bram type=RAM_2P
#pragma HLS ARRAY_PARTITION variable=buffer_q type=complete dim=1
#pragma HLS BIND_STORAGE variable=buffer_k impl=bram type=RAM_2P
#pragma HLS ARRAY_PARTITION variable=buffer_k type=complete dim=1
#pragma HLS BIND_STORAGE variable=buffer_v impl=bram type=RAM_2P
#pragma HLS ARRAY_PARTITION variable=buffer_v type=complete dim=1
#pragma HLS BIND_STORAGE variable=buffer_o impl=bram type=RAM_2P
#pragma HLS ARRAY_PARTITION variable=buffer_o type=complete dim=1
#pragma HLS BIND_STORAGE variable=buffer impl=bram type=RAM_2P
#pragma HLS ARRAY_PARTITION variable=buffer type=complete dim=1

    /*compute ATTN output*/
    Loop_seq:
    for (int i = 0; i < seq_len; i++) {
        if (abs(input[i * d_model] - 0.0) > 1e-7) {
            seq_len_ = i;
        }
    }
    seq_len_ += 1;
    // compute Q = (G1G2G3G4G5G6)q*X1 + bq
    contract(tt_cores_attnq, tt_cores_attnq, buffer_q[0], 0, WD * 1, 0, 1, attn_shape[0], attn_shape[1], tt_ranks[2]); //G1G2, tt_ranks[1]
    contract(buffer_q[0], tt_cores_attnq, buffer_q[1], 0, WD * 2, 0, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3]); //G1G2G3, tt_ranks[2]
    contract_middle(input, buffer_q[1], buffer_q[0], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[0] * attn_shape[1] * attn_shape[2]);//G1G2G3X
    contract(tt_cores_attnq, tt_cores_attnq, buffer[0], WD * 4, WD * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6]); //G5G6, tt_ranks[5]
    contract(tt_cores_attnq, buffer[0], buffer_q[1], WD * 3, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5]); //G4G5G6, tt_ranks[4]
    contract_last(buffer_q[0], buffer_q[1], input_q, 0, 0, 0, seq_len, attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[3]); //G1G2G3XG4G5G6
Loop_Bq_I:
    for (int i = 0; i < seq_len_; i++) {//Wqx+bq
#pragma HLS loop_tripcount max=seq_len
        Loop_Bq_J :
        for (int j = 0; j < d_model * Batchsize; j++) {
#pragma HLS pipeline off
            input_q[i * d_model + j] += bias_q[j];
        }
    }

    // compute K = (G1G2G3G4G5G6)k*X1 + bk
    contract(tt_cores_attnk, tt_cores_attnk, buffer_k[0], 0, WD * 1, 0, 1, attn_shape[0], attn_shape[1], tt_ranks[2]); //G1G2, tt_ranks[1]
    contract(buffer_k[0], tt_cores_attnk, buffer_k[1], 0, WD * 2, 0, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3]); //G1G2G3, tt_ranks[2]
    contract_middle(input, buffer_k[1], buffer_k[0], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[0] * attn_shape[1] * attn_shape[2]);//G1G2G3X
    contract(tt_cores_attnk, tt_cores_attnk, buffer[0], WD * 4, WD * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6]); //G5G6, tt_ranks[5]
    contract(tt_cores_attnk, buffer[0], buffer_k[1], WD * 3, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5]); //G4G5G6, tt_ranks[4]
    contract_last(buffer_k[0], buffer_k[1], input_k, 0, 0, 0, seq_len, attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[3]); //G1G2G3XG4G5G6
Loop_Bk_I:
    for (int i = 0; i < seq_len_; i++) {//Wkx+bk
#pragma HLS loop_tripcount max=seq_len
        Loop_Bk_J:
        for (int j = 0; j < d_model * Batchsize; j++) {
#pragma HLS pipeline off
            input_k[i * d_model + j] += bias_k[j];
        }
    }

    // compute V = (G1G2G3G4G5G6)v*X1 + bv
    contract(tt_cores_attnv, tt_cores_attnv, buffer_v[0], 0, WD * 1, 0, 1, attn_shape[0], attn_shape[1], tt_ranks[2]); //G1G2, tt_ranks[1]
    contract(buffer_v[0], tt_cores_attnv, buffer_v[1], 0, WD * 2, 0, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3]); //G1G2G3, tt_ranks[2]
    contract_middle(input, buffer_v[1], buffer_v[0], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[0] * attn_shape[1] * attn_shape[2]);//G1G2G3X
    contract(tt_cores_attnv, tt_cores_attnv, buffer[0], WD * 4, WD * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6]); //G5G6, tt_ranks[5]
    contract(tt_cores_attnv, buffer[0], buffer_v[1], WD * 3, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5]); //G4G5G6, tt_ranks[4]
    contract_last(buffer_v[0], buffer_v[1], input_v, 0, 0, 0, seq_len, attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[3]); //G1G2G3XG4G5G6
Loop_Bv_I:
    for (int i = 0; i < seq_len_; i++) {//Wkx+bk
#pragma HLS loop_tripcount max=seq_len
        Loop_Bv_J :
        for (int j = 0; j < d_model * Batchsize; j++) {
#pragma HLS pipeline off
            input_v[i * d_model + j] += bias_v[j];
        }
    }

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
    contract(tt_cores_attnfc, tt_cores_attnfc, buffer_o[0], 0, WD * 1, 0, 1, attn_shape[0], attn_shape[1], tt_ranks[2]); //G1G2, tt_ranks[1]
    contract(buffer_o[0], tt_cores_attnfc, buffer[0], 0, WD * 2, 0, attn_shape[0], attn_shape[1], attn_shape[2], tt_ranks[3]); //G1G2G3, tt_ranks[2]
    contract_middle(input_buffer[0], buffer[0], buffer_o[0], 0, 0, 0, seq_len, tt_ranks[3], attn_shape[0] * attn_shape[1] * attn_shape[2]);//G1G2G3X
    contract(tt_cores_attnfc, tt_cores_attnfc, buffer[1], WD * 4, WD * 5, 0, tt_ranks[4], attn_shape[4], attn_shape[5], tt_ranks[6]); //G5G6, tt_ranks[5]
    contract(tt_cores_attnfc, buffer[1], buffer_o[1], WD * 3, 0, 0, tt_ranks[3], attn_shape[3], attn_shape[4], attn_shape[5]); //G4G5G6, tt_ranks[4]
    contract_last(buffer_o[0], buffer_o[1], input_o, 0, 0, 0, seq_len, attn_shape[3], attn_shape[4], attn_shape[5], tt_ranks[3]); //G1G2G3XG4G5G6
Loop_Bo_I:
    for (int i = 0; i < seq_len_; i++) {//Wox+bo+x
#pragma HLS loop_tripcount max=seq_len
        Loop_Bo_J :
        for (int j = 0; j < d_model * Batchsize; j++) {
            input_o[i * d_model + j] += bias_o[j] + input[i * d_model + j];
        }
    }

    // compute dy*dLN(y2)/dy2
    LayerNorm_derivative(input_o, grad_output, grad_output_o, layernorm_w, layernorm_b, seq_len_);

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

    order_control_tt_grad_o(tt_cores_attnfc, tt_ranks, attn_shape, grad_output_o, grad_cores_attnfc, buffer_o, buffer); //G1G2G3X; G4G5G6
    order_control_tt_grad(tt_cores_attnq, tt_ranks, attn_shape, grad_output_q, grad_cores_attnq, buffer_q, buffer); 
    order_control_tt_grad(tt_cores_attnk, tt_ranks, attn_shape, grad_output_k, grad_cores_attnk, buffer_k, buffer);
    order_control_tt_grad(tt_cores_attnv, tt_ranks, attn_shape, grad_output_v, grad_cores_attnv, buffer_v, buffer);
}
