#ifndef ORDER_CONTROL
#define ORDER_CONTROL

#include "defines.h"
#include "layernorm.h"
#include "GELU.h"
/*
void order_control(
    TYPE_WEIGHT* tt_cores_ff1,
    TYPE_WEIGHT* tt_cores_ff2,
    TYPE_WEIGHT* bias_ff1,
    TYPE_WEIGHT* bias_ff2,
    TYPE_WEIGHT* layernorm_w,
    TYPE_WEIGHT* layernorm_b,
    int tt_ranks[6],
    int pff_shape_0[6],
    int pff_shape_1[6]
);*/

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
);


#endif 
