#ifndef ORDER_CONTROL
#define ORDER_CONTROL

#include "defines.h"
#include "contract.h"
#include "layernorm.h"
#include "activation.h"
#include "transpose.h"

void order_control_tt_grad_attn(
    TYPE_WEIGHT* tt_cores,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    TYPE_WEIGHT* grad_cores,
    float buffer_left[2][10 * 12 * 8 * 8],
    float buffer_right[2][10 * 12 * 8 * 8]
);

void order_control_tt_grad_ff1(
    TYPE_WEIGHT* tt_cores_ff1,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    float grad_cores_1[num_cores * WD_FFN],
    float buffer_w1[2][d_model * 10],
    float buffer_w2[2][d_hidden * 10]
);

void order_control_tt_grad_ff2(
    TYPE_WEIGHT* tt_cores_ff2,
    TYPE_DATA* input,
    TYPE_DATA* grad_output,
    float grad_cores_2[num_cores * WD_FFN],
    float buffer_w4[2][d_hidden * 10],
    float buffer_w3[2][d_model * 10]
);

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
    );


#endif 
