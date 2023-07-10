#include "order_control.h"

/*
#include <fstream>
#include <cstring>
#include <string>
*/

using namespace std;
/*
shape = [16,8,8,8,8,16]
rank = [1,16,30,30,30,16,1]
*/
float input[seq_len * 12 * 8 * 8 * Batchsize]{0};
float grad_output[seq_len * 12 * 8 * 8 * Batchsize]{0};
float input_2[seq_len * 16 * 16 * 12 * Batchsize]{ 0 };
float dgelu[seq_len * 16 * 16 * 12 * Batchsize]{ 0 };
float output_2[seq_len * 8 * 8 * 12 * Batchsize]{ 0 };
float grad_output1[seq_len * 12 * 16 * 16 * Batchsize]{ 0 };
float grad_output2[seq_len * 12 * 8 * 8 * Batchsize]{ 0 };
//float grad_cores_1[num_cores * WD]{0};
//float grad_cores_2[num_cores * WD]{0};

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
                    output[i * J * K * L + j * K * L + k * L + l + o_offset] = res;//IJKL
                }
            }
        }
    }
}// tensor contract to 2-D

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
                    output[i * J * K * L + j * K * L + k * L + l + o_offset] = res;//IJKL
                }
            }
        }
    }
} // tensor contract to 3-D


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
} // tensor contract to 1-D

void order_control(
    TYPE_WEIGHT* tt_cores_ff1,
    TYPE_WEIGHT* tt_cores_ff2,
    TYPE_WEIGHT* bias_ff1,
    TYPE_WEIGHT* bias_ff2,
    TYPE_WEIGHT* layernorm_w,
    TYPE_WEIGHT* layernorm_b,
    int tt_ranks[6],
    int pff_shape_0[6],
    int pff_shape_1[6],
    float grad_cores_1[num_cores * WD],
    float grad_cores_2[num_cores * WD]
) {
    float buffer_w2[2][10 * 12 * 16 * 16]{ 0 };
    float buffer_w4[2][10 * 12 * 16 * 16]{ 0 };
    float buffer_w1[2][10 * 12 * 8 * 8]{ 0 };
    float buffer_w3[2][10 * 12 * 8 * 8]{ 0 };
    // compute x2 = GELU(y1) = GELU(W1x1+b1) = GELU((G1G2G3xG4G5G6)2+b1)
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w1[0], 0, WD * 1, 0, 1, pff_shape_0[0], pff_shape_0[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract(buffer_w1[0], tt_cores_ff1, buffer_w1[1], 0, WD * 2, 0, pff_shape_0[0], pff_shape_0[1], pff_shape_0[2], tt_ranks[3], tt_ranks[2]); //G1G2G3
    contract_middle(input, buffer_w1[1], buffer_w1[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_0[0] * pff_shape_0[1] * pff_shape_0[2]);//G1G2G3X
    contract(tt_cores_ff1, tt_cores_ff1, buffer_w2[0], WD * 4, WD * 5, 0, tt_ranks[4], pff_shape_0[4], pff_shape_0[5], tt_ranks[6], tt_ranks[5]); //G5G6
    contract(tt_cores_ff1, buffer_w2[0], buffer_w2[1], WD * 3, 0, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], pff_shape_0[5], tt_ranks[4]); //G4G5G6
    contract_last(buffer_w1[0], buffer_w2[1], input_2, 0, 0, 0, seq_len, pff_shape_0[3], pff_shape_0[4], pff_shape_0[5], tt_ranks[3]); //G1G2G3XG4G5G6
    for (int i = 0; i < seq_len; i++) {//W1x1+b1
        for (int j = 0; j < pff_shape_0[3] * pff_shape_0[4] * pff_shape_0[5] * Batchsize; j++) {
            input_2[i * d_hidden + j] += bias_ff1[j];
        }
    }
    gelu(input_2, seq_len * pff_shape_0[3] * pff_shape_0[4] * pff_shape_0[5] * Batchsize); //x2 = GELU(W1X1+b1)

    // compute y2 = W2x2+b2 = (G1G2G3x2G4G5G6)2 + b2
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[0], 0, WD * 1, 0, 1, pff_shape_1[0], pff_shape_1[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract(buffer_w4[0], tt_cores_ff2, buffer_w4[1], 0, WD * 2, 0, pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], tt_ranks[3], tt_ranks[2]); //G1G2G3
    contract_middle(input_2, buffer_w4[1], buffer_w4[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_1[0] * pff_shape_1[1] * pff_shape_1[2]);//G1G2G3X
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[0], WD * 4, WD * 5, 0, tt_ranks[4], pff_shape_1[4], pff_shape_1[5], tt_ranks[6], tt_ranks[5]); //G5G6
    contract(tt_cores_ff2, buffer_w3[0], buffer_w3[1], WD * 3, 0, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], tt_ranks[4]); //G4G5G6
    contract_last(buffer_w4[0], buffer_w3[1], output_2, 0, 0, 0, seq_len, pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], tt_ranks[3]); //G1G2G3XG4G5G6
    for (int i = 0; i < seq_len; i++) {//W2x2+b2
        for (int j = 0; j < pff_shape_1[3] * pff_shape_1[4] * pff_shape_1[5] * Batchsize; j++) {
            output_2[i * d_model + j] += bias_ff2[j];
        }
    }

    // compute grad_output2 = dy*dLN(y2)/dy2
    LayerNorm_derivative(output_2, grad_output2, layernorm_w, layernorm_b, seq_len, d_model);//dLN(y2)/dy2
    for (int i = 0; i < seq_len * 12 * 8 * 8 * Batchsize; i++) {//dy2*dLN(y2)/dy2
        grad_output2[i] *= grad_output[i];
    }

    // compute grad_output1 = dy*dLN(y2)/dy2*(G4G5G6)2*(G1G2G3)2*dGELU(y1)/dy1
    contract_middle(grad_output2, buffer_w3[1], buffer_w3[0], 0, 0, 0, seq_len, tt_ranks[3], pff_shape_1[3] * pff_shape_1[4] * pff_shape_1[5]);//dy2*dLN(y2)/dy2*(G4G5G6)2
    contract_last(buffer_w3[0], buffer_w4[1], grad_output1, 0, 0, 0, seq_len, pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], tt_ranks[3]);//dy2*dLN(y2)/dy2*(G4G5G6)2*(G1G2G3)2

    gelu_derivative(input_2, dgelu, seq_len * pff_shape_0[3] * pff_shape_0[4] * pff_shape_0[5] * Batchsize);//dGELU(y1)/dy1
    for (int i = 0; i < seq_len * pff_shape_0[3] * pff_shape_0[4] * pff_shape_0[5] * Batchsize; i++) { //dy*dLN(y2)/dy2*(G4G5G6)2*(G1G2G3)2*dGELU(y1)/dy1
        grad_output1[i] *= dgelu[i];
    }

    //FILE* file1;
    //fopen_s(&file1, "grad1.txt", "w");
    //FILE* file2;
    //fopen_s(&file2, "grad2.txt", "w");
    //for (int i = 0; i < seq_len*d_model; i++) {
	   // fprintf(file2, "%f ", grad_output2[i]);
	   // }
    //for (int i = 0; i < seq_len * d_hidden; i++) {
    //    fprintf(file1, "%f ", grad_output1[i]);
    //}
    //fclose(file1);
    //fclose(file2);
    //compute grad cores
    //order_control_tt_grad(tt_cores_ff1, tt_ranks, pff_shape_0, input, grad_output1, grad_cores_ff1, buffer_w1, buffer_w2); //[0]G1G2G3X,[1]G1G2G3;[0]G5G6,[1]G4G5G6;
    //order_control_tt_grad(tt_cores_ff2, tt_ranks, pff_shape_1, input_2, grad_output2, grad_cores_ff2, buffer_w4, buffer_w3); //[0]G1G2G3X,[1]G1G2G3;[0]xxx,G4G5G6;

    //compute grad cores for W2
    contract_last(buffer_w4[0], grad_output2, buffer_w3[0], 0, 0, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], pff_shape_1[5], seq_len);//(G1G2G3X)dy
    contract_middle(grad_output2, buffer_w3[1], buffer_w4[1], 0, 0, 0, seq_len, tt_ranks[4], pff_shape_1[3]* pff_shape_1[4]* pff_shape_1[5]);//(G4G5G6dy)
    contract_last(buffer_w4[1], input_2, buffer_w4[0], 0, 0, 0, tt_ranks[4], pff_shape_1[0], pff_shape_1[1], pff_shape_1[2], seq_len);//(G4G5G6dy)X
    //gradient_6
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[1], WD * 3, WD * 4, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[4], tt_ranks[5], tt_ranks[4]); //G4G5
    contract_middle(buffer_w3[0], buffer_w3[1], grad_cores_2, 0, 0, WD * 5, pff_shape_1[5], tt_ranks[5], tt_ranks[3] * pff_shape_1[3] * pff_shape_1[4]);//(G1G2G3X)dyG4G5=gradient_6
    //gradient_5
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[1], WD * 3, WD * 5, 0, tt_ranks[3], pff_shape_1[3], pff_shape_1[5], tt_ranks[6], tt_ranks[5]); //G4G6
    contract(buffer_w3[0], buffer_w3[1], grad_cores_2, 0, 0, WD * 4, tt_ranks[4], pff_shape_1[4], tt_ranks[5], tt_ranks[6], pff_shape_1[3] * pff_shape_1[5]);//(G1G2G3X)dyG4G6=gradient_5                                                                                                                                                
    //gradient_4
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w3[1], WD * 4, WD * 5, 0, tt_ranks[4], pff_shape_1[4], pff_shape_1[5], tt_ranks[6], tt_ranks[5]); //G5G6
    contract(buffer_w3[0], buffer_w3[1], grad_cores_2, 0, 0, WD * 3, tt_ranks[3], pff_shape_1[3], tt_ranks[4], tt_ranks[6], pff_shape_1[4] * pff_shape_1[5]);//(G1G2G3X)dyG5G6=gradient_4                                                                                                                                               
    //gradient_3
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[1], 0, WD * 1, 0, tt_ranks[0], pff_shape_1[0], pff_shape_1[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract_middle(buffer_w4[0], buffer_w4[1], grad_cores_2, 0, 0, WD * 2, tt_ranks[2], pff_shape_1[2], tt_ranks[0] * pff_shape_1[0] * pff_shape_1[1]);//(G4G5G6dy)XG1G2=gradient_3
    //gradient_2
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[1], 0, WD * 2, 0, tt_ranks[0], pff_shape_1[0], pff_shape_1[2], tt_ranks[3], tt_ranks[2]); //G1G3
    contract(buffer_w4[0], buffer_w4[1], grad_cores_2, 0, 0, WD * 1, tt_ranks[1], pff_shape_1[1], tt_ranks[2], 1, pff_shape_1[0] * pff_shape_1[2]);//(G4G5G6dy)XG1G3=gradient_2                                                                                                                                                
    //gradient_1
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w4[1], WD * 1, WD * 2, 0, tt_ranks[1], pff_shape_1[1], pff_shape_1[2], tt_ranks[3], tt_ranks[2]); //G2G3
    contract(buffer_w4[0], buffer_w4[1], grad_cores_2, 0, 0, 0, tt_ranks[0], pff_shape_1[0], tt_ranks[1], 1, pff_shape_1[1] * pff_shape_1[2]);//(G4G5G6dy)XG2G3=gradient_1                                                                                                                                          
    
    //compute grad cores for W1
    contract_last(buffer_w1[0], grad_output1, buffer_w2[0], 0, 0, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], pff_shape_0[5], seq_len);//(G1G2G3X)dy
    contract_middle(buffer_w2[1], grad_output1, buffer_w1[1], 0, 0, 0, tt_ranks[4], seq_len, pff_shape_0[3] * pff_shape_0[4] * pff_shape_0[5]);//(G4G5G6dy)
    contract_last(buffer_w1[1], input, buffer_w1[0], 0, 0, 0, tt_ranks[4], pff_shape_0[0], pff_shape_0[1], pff_shape_0[2], seq_len);//(G4G5G6dy)X
    //gradient_6
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w2[1], WD * 3, WD * 4, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[4], tt_ranks[5], tt_ranks[4]); //G4G5
    contract_middle(buffer_w2[0], buffer_w2[1], grad_cores_1, 0, 0, WD * 5, pff_shape_0[5], tt_ranks[5], tt_ranks[3] * pff_shape_0[3] * pff_shape_0[4]);//(G1G2G3X)dyG4G5=gradient_6
    //gradient_5
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w2[1], WD * 3, WD * 5, 0, tt_ranks[3], pff_shape_0[3], pff_shape_0[5], tt_ranks[6], tt_ranks[5]); //G4G6
    contract(buffer_w2[0], buffer_w2[1], grad_cores_1, 0, 0, WD * 4, tt_ranks[4], pff_shape_0[4], tt_ranks[5], tt_ranks[6], pff_shape_0[3] * pff_shape_0[5]);//(G1G2G3X)dyG4G6=gradient_5                                                                                                                                                
    //gradient_4
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w2[1], WD * 4, WD * 5, 0, tt_ranks[4], pff_shape_0[4], pff_shape_0[5], tt_ranks[6], tt_ranks[5]); //G5G6
    contract(buffer_w2[0], buffer_w2[1], grad_cores_1, 0, 0, WD * 3, tt_ranks[3], pff_shape_0[3], tt_ranks[4], tt_ranks[6], pff_shape_0[4] * pff_shape_0[5]);//(G1G2G3X)dyG5G6=gradient_4                                                                                                                                               
    //gradient_3
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w1[1], 0, WD * 1, 0, tt_ranks[0], pff_shape_0[0], pff_shape_0[1], tt_ranks[2], tt_ranks[1]); //G1G2
    contract_middle(buffer_w1[0], buffer_w1[1], grad_cores_1, 0, 0, WD * 2, tt_ranks[2], pff_shape_0[2], tt_ranks[0] * pff_shape_0[0] * pff_shape_0[1]);//(G4G5G6dy)XG1G2=gradient_3
    //gradient_2
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w1[1], 0, WD * 2, 0, tt_ranks[0], pff_shape_0[0], pff_shape_0[2], tt_ranks[3], tt_ranks[2]); //G1G3
    contract(buffer_w1[0], buffer_w1[1], grad_cores_1, 0, 0, WD * 1, tt_ranks[1], pff_shape_0[1], tt_ranks[2], 1, pff_shape_0[0] * pff_shape_0[2]);//(G4G5G6dy)XG1G3=gradient_2                                                                                                                                                
    //gradient_1
    contract(tt_cores_ff2, tt_cores_ff2, buffer_w1[1], WD * 1, WD * 2, 0, tt_ranks[1], pff_shape_0[1], pff_shape_0[2], tt_ranks[3], tt_ranks[2]); //G2G3
    contract(buffer_w1[0], buffer_w1[1], grad_cores_1, 0, 0, 0, tt_ranks[0], pff_shape_0[0], tt_ranks[1], 1, pff_shape_0[1] * pff_shape_0[2]);//(G4G5G6dy)XG2G3=gradient_1 

}
