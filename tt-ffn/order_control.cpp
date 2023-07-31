#include "order_control.h"
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
/*
shape = [16,8,8,8,8,16]
rank = [1,16,30,30,30,16,1]
*/
#ifdef __SYNTHESIS__
float input[seq_len * 12 * 8 * 8 * Batchsize]{0};
float grad_output[seq_len * 12 * 8 * 8 * Batchsize]{0};
float grad_cores_1[num_cores * WD]{ 0 };
float grad_cores_2[num_cores * WD]{ 0 };
#else
extern float input[seq_len * 12 * 8 * 8 * Batchsize];
extern float grad_output[seq_len * 12 * 8 * 8 * Batchsize];
extern float grad_cores_1[num_cores * WD];
extern float grad_cores_2[num_cores * WD];
#endif
float input_2[seq_len * 16 * 16 * 12 * Batchsize]{ 0 };
float output_1[seq_len * 16 * 16 * 12 * Batchsize]{ 0 };
float output_2[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float grad_output1[seq_len * 12 * 16 * 16 * Batchsize]{ 0 };
float grad_output2[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };


int seq_len_;

void order_control_ff1(
    TYPE_WEIGHT* tt_cores_ff1,
    int tt_ranks[7],
    int pff_shape_0[6],
    float buffer_w1[2][d_model*10],
    float buffer_w2[2][d_hidden*10]
) {
    //compute grad cores for W1
    transpose(buffer_w1[0], buffer_w1[1], seq_len, 10);// 10,32
    contract_last(buffer_w1[1], grad_output1, buffer_w2[0], 0, 0, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], pff_shape_0[5], seq_len);//(G1G2G3X)dy [10,12,16,16,32]
    clear(buffer_w1[1], d_model * 10);
    contract_middle_right(grad_output1, buffer_w2[1], buffer_w1[1], 0, 0, 0, seq_len, tt_ranks[4], pff_shape_0[3] * pff_shape_0[4] * pff_shape_0[5]);//(G4G5G6dy) 10 [32,10,16*16*12]
    contract_last_left(buffer_w1[1], input, buffer_w1[0], 0, 0, 0, tt_ranks[4], pff_shape_0[0], pff_shape_0[1], pff_shape_0[2], seq_len);//(G4G5G6dy)X [10,12,8,8,32]
    transpose(buffer_w1[0], buffer_w1[1], tt_ranks[4], pff_shape_0[0] * pff_shape_0[1] * pff_shape_0[2]);//3072, 10
    //gradient_6
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w2[1], WD * 3, WD * 4, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], tt_ranks[5], tt_ranks[4]); //G4G5 [10,12,16,10]
    contract_middle_left(buffer_w2[1], buffer_w2[0], grad_cores_1, 0, 0, WD * 5, tt_ranks[3], pff_shape_0[5], tt_ranks[5] * pff_shape_0[3] * pff_shape_0[4]);//(G1G2G3X)dyG4G5=gradient_6 [10,16,10*12*16]
    //gradient_4
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w2[1], WD * 4, WD * 5, 0, tt_ranks[4], pff_shape_0[4], pff_shape_0[5], tt_ranks[6], tt_ranks[5]); //G5G6 [10,16,16,1]
    contract_right(buffer_w2[0], buffer_w2[1], grad_cores_1, 0, 0, WD * 3, tt_ranks[3],  pff_shape_0[3], tt_ranks[4], tt_ranks[6], pff_shape_0[4] * pff_shape_0[5]);//(G1G2G3X)dyG5G6=gradient_4 [10,12,10,1,12*16]
    //gradient_5
    clear(buffer_w2[1], d_hidden * 10);
    contract_left(tt_cores_ff1, buffer_w2[0], buffer_w2[1], WD * 3, 0, 0, tt_ranks[3], 1, pff_shape_0[4], pff_shape_0[5], tt_ranks[4] * pff_shape_0[3]); //(G1G2G3X)dyG4 [10,1,16,16,120]
    contract_right(buffer_w2[1], tt_cores_ff1, grad_cores_1, 0, WD * 5, WD * 4, tt_ranks[4], pff_shape_0[4], tt_ranks[5], tt_ranks[6], pff_shape_0[5]);//(G1G2G3X)dyG4G6=gradient_5 [10,16,10,1,16]

    //gradient_3
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w1[0], 0, WD * 1, 0, tt_ranks[0], pff_shape_0[0], pff_shape_0[1], tt_ranks[2], tt_ranks[1]); //G1G2 [1,12,8,10]
    contract_left(buffer_w1[0], buffer_w1[1], grad_cores_1, 0, 0, WD * 2, tt_ranks[0], tt_ranks[2], pff_shape_0[2], tt_ranks[3], pff_shape_0[0] * pff_shape_0[1]);//(G4G5G6dy)XG1G2=gradient_3 [1,10,8,10,96]
    //gradient_1
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w1[0], WD * 1, WD * 2, 0, tt_ranks[1], pff_shape_0[1], pff_shape_0[2], tt_ranks[3], tt_ranks[2]); //G2G3 [10,8,8,10]
    contract_middle_right(buffer_w1[1], buffer_w1[0], grad_cores_1, 0, 0, 0, pff_shape_0[0], tt_ranks[1], tt_ranks[3] * pff_shape_0[1] * pff_shape_0[2]);//(G4G5G6dy)XG2G3=gradient_1 [12,10,10*8*8]
    //gradient_2
    contract_last_left(tt_cores_ff1, buffer_w1[1], buffer_w1[0], 0, 0, 0, tt_ranks[1], pff_shape_0[1], pff_shape_0[2], tt_ranks[2], pff_shape_0[0]);//(G4G5G6dy)XG1 [10,8,8,10,12]
    contract_right(buffer_w1[0], tt_cores_ff1, grad_cores_1, 0, WD * 2, WD * 1, tt_ranks[1], pff_shape_0[1], tt_ranks[0], tt_ranks[3], tt_ranks[2] * pff_shape_0[2]);//(G4G5G6dy)XG1G3=gradient_2 [10,8,1,10,10*8]
}

void order_control_ff2(
    TYPE_WEIGHT* tt_cores_ff2,
    int tt_ranks[7],
    int pff_shape_1[6],
    float buffer_w4[2][d_hidden * 10],
    float buffer_w3[2][d_model * 10]
) {
    //compute grad cores for W2
    transpose(buffer_w4[0], buffer_w4[1], seq_len, 10);// 10,32
    contract_last(buffer_w4[1], grad_output2, buffer_w3[0], 0, 0, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], seq_len);//(G1G2G3X)dy [10,8,8,12,32]
    clear(buffer_w4[1], d_hidden * 10);
    contract_middle_right(grad_output2, buffer_w3[1], buffer_w4[1], 0, 0, 0, seq_len, tt_ranks[4], pff_shape_1[3] * pff_shape_1[4] * pff_shape_1[5]);//(G4G5G6dy)
    contract_last_left(buffer_w4[1], input_2, buffer_w4[0], 0, 0, 0, tt_ranks[4],pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], seq_len);//(G4G5G6dy)X [10,16,16,12,32]
    transpose(buffer_w4[0], buffer_w4[1], tt_ranks[4], pff_shape_1[0] * pff_shape_1[1] * pff_shape_1[2]);//3072, 10
    //gradient_6
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[1], WD * 3, WD * 4, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], tt_ranks[5], tt_ranks[4]); //G4G5 10,8,8,10
    contract_middle_left(buffer_w3[1], buffer_w3[0], grad_cores_2, 0, 0, WD * 5, tt_ranks[3], pff_shape_1[5], tt_ranks[5] * pff_shape_1[3] * pff_shape_1[4]);//(G1G2G3X)dyG4G5=gradient_6 [10,12,10*8*8]
    //gradient_4
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[1], WD * 4, WD * 5, 0, tt_ranks[4], pff_shape_1[4], pff_shape_1[5], tt_ranks[6], tt_ranks[5]); //G5G6 10,8,8,1
    contract_right(buffer_w3[0], buffer_w3[1], grad_cores_2, 0, 0, WD * 3, tt_ranks[3], pff_shape_1[3], tt_ranks[4], tt_ranks[6], pff_shape_1[4] * pff_shape_1[5]);//(G1G2G3X)dyG5G6=gradient_4
    //gradient_5
    clear(buffer_w3[1], d_model * 10);
    contract_left(tt_cores_ff2, buffer_w3[0], buffer_w3[1], WD * 3, 0, 0, tt_ranks[3], 1, pff_shape_1[4], pff_shape_1[5], tt_ranks[4] * pff_shape_1[3]); //(G1G2G3X)dyG4 [10,1,8,8,96]
    contract_right(buffer_w3[1], tt_cores_ff2, grad_cores_2, 0, WD*5, WD * 4, tt_ranks[4], pff_shape_1[4], tt_ranks[5], tt_ranks[6], pff_shape_1[5]);//(G1G2G3X)dyG4G6=gradient_5 [10,8,10,1,12]

    //gradient_3
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[0], 0, WD * 1, 0, tt_ranks[0], pff_shape_1[0], pff_shape_1[1], tt_ranks[3], tt_ranks[2]); //G1G2 1,16,16,10, tt_ranks[1]
    contract_left(buffer_w4[0], buffer_w4[1], grad_cores_2, 0, 0, WD * 2, tt_ranks[0], tt_ranks[2], pff_shape_1[2], tt_ranks[3], pff_shape_1[0] * pff_shape_1[1]);//(G4G5G6dy)XG1G2=gradient_3 [1,10,12,10,256]
    //gradient_1
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[0], WD * 1, WD * 2, 0, tt_ranks[1], pff_shape_1[1], pff_shape_1[2], tt_ranks[3], tt_ranks[2]); //G2G3
    contract_middle_right(buffer_w4[1], buffer_w4[0], grad_cores_2, 0, 0, 0, pff_shape_1[0], tt_ranks[1], tt_ranks[3] * pff_shape_1[1] * pff_shape_1[2]);//(G4G5G6dy)XG2G3=gradient_1 [16,10,10*16*12]
    //gradient_2
    contract_last_left(tt_cores_ff2, buffer_w4[1], buffer_w4[0], 0, 0, 0, tt_ranks[1], pff_shape_1[1], pff_shape_1[2], tt_ranks[2], pff_shape_1[0]);//(G4G5G6dy)XG1 [10,16,12,10,16]
    contract_right(buffer_w4[0], tt_cores_ff2, grad_cores_2, 0, WD*2, WD * 1, tt_ranks[1], pff_shape_1[1], tt_ranks[0], tt_ranks[3], tt_ranks[2] * pff_shape_1[2]);//(G4G5G6dy)XG1G3=gradient_2 [10,16,1,10,10*12]
}

void order_control_top(
    TYPE_WEIGHT* tt_cores_ff1,
    TYPE_WEIGHT* tt_cores_ff2,
    TYPE_WEIGHT* bias_ff1,
    TYPE_WEIGHT* bias_ff2,
    TYPE_WEIGHT* layernorm_w,
    TYPE_WEIGHT* layernorm_b,
    int tt_ranks[7],
    int pff_shape_0[6],
    int pff_shape_1[6]
) {
    float buffer_w2[2][10 * 12 * 16 * 16]{ 0 };
    float buffer_w4[2][10 * 12 * 16 * 16]{ 0 };
    float buffer_w1[2][10 * 12 * 8 * 8]{ 0 };
    float buffer_w3[2][10 * 12 * 8 * 8]{ 0 };

#pragma HLS ARRAY_PARTITION variable=buffer_w1 type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_w2 type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_w3 type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer_w4 type=complete dim=1
#pragma HLS interface mode=bram port=tt_cores_ff1
#pragma HLS interface mode=bram port=tt_cores_ff2
#pragma HLS interface mode=bram port=bias_ff1
#pragma HLS interface mode=bram port=bias_ff2
#pragma HLS interface mode=bram port=layernorm_w
#pragma HLS interface mode=bram port=layernorm_b

    Loop_seq:
    for (int i = 0; i < seq_len; i++) {
        if (abs(input[i * d_model] - 0.0) > 1e-7) {
            seq_len_ = i;
        }
    }
    seq_len_ += 1;
    // compute x2 = GELU(y1) = GELU(W1x1+b1) = GELU((G1G2G3xG4G5G6)2+b1)
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w1[0], 0, WD * 1, 0, tt_ranks[0], pff_shape_0[0], pff_shape_0[1], tt_ranks[2], tt_ranks[1]); //G1G2 [1,12,8,10] 
    contract(buffer_w1[0], tt_cores_ff1, buffer_w1[1], 0, WD * 2, 0, pff_shape_0[0], pff_shape_0[1], pff_shape_0[2], tt_ranks[3], tt_ranks[2]); //G1G2G3 [12,8,8,10]
    contract_middle(input, buffer_w1[1], buffer_w1[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_0[0] * pff_shape_0[1] * pff_shape_0[2]);//G1G2G3X [32,10,12*8*8]
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w2[0], WD * 4, WD * 5, 0, tt_ranks[4], pff_shape_0[4], pff_shape_0[5], tt_ranks[6], tt_ranks[5]); //G5G6 [10,16,16,1]
    contract(tt_cores_ff1, buffer_w2[0], buffer_w2[1], WD * 3, 0, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], pff_shape_0[5],tt_ranks[4]); //G4G5G6 [10,12,16,16] 
    contract_last(buffer_w1[0], buffer_w2[1], output_1, 0, 0, 0, seq_len, pff_shape_0[3], pff_shape_0[4], pff_shape_0[5], tt_ranks[3]); //G1G2G3XG4G5G6 [32,12,16,16,10]
    Loop_B1_I:
    for (int i = 0; i < seq_len_; i++) {//W1x1+b1
#pragma HLS loop_tripcount max=seq_len
        Loop_B1_J:
        for (int j = 0; j < d_hidden * Batchsize; j++) {
            output_1[i * d_hidden + j] += bias_ff1[j];
        }
    }
	Loop_gelu:
	for (int i = 0; i < seq_len_ * d_hidden * Batchsize; i++) {
#pragma HLS loop_tripcount max=seq_len
		input_2[i] = 0.5 * output_1[i] * (1.0 + tanhf(sqrtf(2.0 / M_PI) * (output_1[i] + 0.044715 * powf(output_1[i], 3))));
	}

    // compute y2 = W2x2+b2 = (G1G2G3x2G4G5G6)2 + b2
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[0], 0, WD * 1, 0, tt_ranks[0], pff_shape_1[0], pff_shape_1[1], tt_ranks[2], tt_ranks[1]); //G1G2 1,16,16,10,10
    contract(buffer_w4[0], tt_cores_ff2, buffer_w4[1], 0, WD * 2, 0, pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], tt_ranks[3], tt_ranks[2]); //G1G2G3 16,16,12,10,10
    contract_middle(input_2, buffer_w4[1], buffer_w4[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_1[0] * pff_shape_1[1] * pff_shape_1[2]);//G1G2G3X [32,10,16*16*12]
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[0], WD * 4, WD * 5, 0, tt_ranks[4], pff_shape_1[4], pff_shape_1[5], tt_ranks[6], tt_ranks[5]); //G5G6 10,8,12,1,10
    contract(tt_cores_ff2, buffer_w3[0], buffer_w3[1], WD * 3, 0, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], tt_ranks[4]); //G4G5G6 10,8,8,12,10
    contract_last(buffer_w4[0], buffer_w3[1], output_2, 0, 0, 0, seq_len, pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], tt_ranks[3]); //G1G2G3XG4G5G6 [32,8,8,12,10]
    Loop_B2_I:
    for (int i = 0; i < seq_len_; i++) {//W2x2+b2
#pragma HLS loop_tripcount max=seq_len
        Loop_B2_J:
        for (int j = 0; j < d_model * Batchsize; j++) {
            output_2[i * d_model + j] += bias_ff2[j] + input[i * d_model + j];
        }
    }

    // compute grad_output2 = dy*dLN(y2)/dy2
    LayerNorm_derivative(output_2, grad_output, grad_output2, layernorm_w, layernorm_b, seq_len_);

    // compute grad_output1 = dy*dLN(y2)/dy2*(G4G5G6)2*(G1G2G3)2*dGELU(y1)/dy1
    clear(buffer_w3[0], d_model* 10);
    contract_middle_right(grad_output2, buffer_w3[1], buffer_w3[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_1[3] * pff_shape_1[4] * pff_shape_1[5]);//dy2*dLN(y2)/dy2*(G4G5G6)2 (32,768)*(10,768) [32,10,8*8*12]
    contract_last_right(buffer_w3[0], buffer_w4[1], grad_output1, 0, 0, 0, seq_len, pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], tt_ranks[3]);//dy2*dLN(y2)/dy2*(G4G5G6)2*(G1G2G3)2 (32,10)*(16,16,12,10)
	Loop_dgelu:
    for (int i = 0; i < seq_len_ * d_hidden * Batchsize; i++) {
#pragma HLS loop_tripcount max=seq_len * d_hidden
        float cdf = 1.0 + tanhf(sqrtf(2.0 / M_PI) * (output_1[i] + 0.044715 * powf(output_1[i], 3)));
        float pdf = sqrtf(2.0 / M_PI) * expf(-0.5 * output_1[i] * output_1[i]);
        grad_output1[i] *= 0.5 * (cdf + output_1[i] * pdf);
    }

    //compute grad cores
    order_control_ff1(tt_cores_ff1, tt_ranks, pff_shape_0, buffer_w1, buffer_w2); //[0]G1G2G3X,[1]G1G2G3;[0]G5G6,[1]G4G5G6;
    order_control_ff2(tt_cores_ff2, tt_ranks, pff_shape_1, buffer_w4, buffer_w3); //[0]G1G2G3X,[1]G1G2G3;[0]xxx,G4G5G6;
}
