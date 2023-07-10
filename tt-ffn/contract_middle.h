#ifndef CONTRACT_MIDDLE
#define CONTRACT_MIDDLE

#include "defines.h"
void contract_middle(
    float* tensor_l,
    float* tensor_r,
    float* output,
    int l_offset,
    int r_offset,
    int o_offset,
    int I,
    int R,
    int M);
#endif

#pragma once
