#include <iostream>
#include "defines.h"

void clear(
    float* input,
    int size
) {
LOOP_CLEAR:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount max=10*d_model
        input[i] = 0;
    }
}
