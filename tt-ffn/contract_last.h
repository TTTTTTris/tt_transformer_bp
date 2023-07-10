#ifndef CONTRACT_LAST
#define CONTRACT_LAST

#include "defines.h"

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

#endif

