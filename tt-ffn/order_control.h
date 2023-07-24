#ifndef ORDER_CONTROL
#define ORDER_CONTROL

#include "defines.h"
#include "layernorm.h"
#include "contract.h"
//#include "GELU.h"
#include "transpose.h"

void order_control_ff1(
    TYPE_WEIGHT* tt_cores_ff1,
    int tt_ranks[7],
    int pff_shape_0[6],
    float grad_cores_1[num_cores * WD],
    float buffer_w1[2][d_model * 10],
    float buffer_w2[2][d_hidden * 10]
);

void order_control_ff2(
    TYPE_WEIGHT* tt_cores_ff2,
    int tt_ranks[7],
    int pff_shape_1[6],
    float grad_cores_2[num_cores * WD],
    float buffer_w4[2][d_hidden * 10],
    float buffer_w3[2][d_model * 10]
);

void order_control_top(
    TYPE_WEIGHT* tt_cores_ff1,
    TYPE_WEIGHT* tt_cores_ff2,
    TYPE_WEIGHT* bias_ff1,
    TYPE_WEIGHT* bias_ff2,
    TYPE_WEIGHT* layernorm_w,
    TYPE_WEIGHT* layernorm_b,
    int tt_ranks[7],
    int pff_shape_0[6],
    int pff_shape_1[6],
    float grad_cores_1[num_cores * WD],
    float grad_cores_2[num_cores * WD]
);


#endif 
