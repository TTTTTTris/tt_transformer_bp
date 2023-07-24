#include <iostream>
#include "transpose.h"

void transpose(float* input, float* output, int I, int J) {
	LOOP_I:
	for (int i = 0; i < I; ++i) {
#pragma HLS loop_tripcount min=10 max=seq_len
		LOOP_J:
		for (int j = 0; j < J; ++j) {
#pragma HLS loop_tripcount min=10 max=d_hidden
				output[j * I + i] = input[i * J + j]; //IJK->JIK
		}
	}
}
