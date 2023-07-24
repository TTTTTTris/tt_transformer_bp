#ifndef CONTRACT
#define CONTRACT

#include "defines.h"

void clear(
    float* input,
    int size
);

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
    int L
    //const int R
);

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
);

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
);


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
);

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
);

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
);

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
);


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
);

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
);
#endif
#pragma once

