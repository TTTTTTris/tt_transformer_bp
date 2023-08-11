#ifndef ORDER_CONTROL
#define ORDER_CONTROL

#include "defines.h"
#include "contract.h"
#include "matrix.h"
#include "layernorm.h"
#include "GELU.h"
#include "softmax.h"
#include "transpose.h"

void order_control_tt_grad(
    TYPE_WEIGHT* tt_cores,
    int* tt_ranks,
    int* tt_shapes,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    TYPE_WEIGHT* grad_cores,
    float buffer_left[2][10 * 12 * 8 * 8],
    float buffer_right[2][10 * 12 * 8 * 8]
);

void order_control_tt_grad_fc(
    TYPE_WEIGHT* tt_cores_ff1,
    TYPE_WEIGHT* tt_cores_ff2,
    TYPE_WEIGHT* bias_ff1,
    TYPE_WEIGHT* bias_ff2,
    TYPE_WEIGHT* layernorm_w,
    TYPE_WEIGHT* layernorm_b,
    int tt_ranks[6],
    int pff_shape_0[6],
    int pff_shape_1[6],
    TYPE_DATA* input,
    TYPE_DATA* input_2,
    TYPE_DATA* grad_output1,
    TYPE_DATA* grad_output2,
    TYPE_WEIGHT* grad_cores_ff1,
    TYPE_WEIGHT* grad_cores_ff2,
    float buffer_w1[][10 * 12 * 8 * 8],
    float buffer_w2[][10 * 12 * 16 * 16],
    float buffer_w3[][10 * 12 * 8 * 8],
    float buffer_w4[][10 * 12 * 16 * 16]
);

//void order_control_tt_grad_attn(
//    TYPE_WEIGHT* tt_cores_attnq,
//    TYPE_WEIGHT* tt_cores_attnk,
//    TYPE_WEIGHT* tt_cores_attnv,
//    TYPE_WEIGHT* tt_cores_attnfc,
//    TYPE_WEIGHT* bias_q,
//    TYPE_WEIGHT* bias_k,
//    TYPE_WEIGHT* bias_v,
//    TYPE_WEIGHT* bias_o,
//    TYPE_WEIGHT* layernorm_w,
//    TYPE_WEIGHT* layernorm_b,
//    int* tt_ranks,
//    int* attn_shape,
//    TYPE_DATA* input,
//    TYPE_DATA* grad_output,
//    TYPE_WEIGHT* grad_cores_attnq,
//    TYPE_WEIGHT* grad_cores_attnk,
//    TYPE_WEIGHT* grad_cores_attnv,
//    TYPE_WEIGHT* grad_cores_attnfc
//);

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
        TYPE_WEIGHT* fc_layernorm_b,
        int tt_ranks[6],
        int attn_shape[6],
        int pff_shape_0[6],
        int pff_shape_1[6]
    );


#endif 
