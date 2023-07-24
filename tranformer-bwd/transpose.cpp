#include <iostream>
#include "transpose.h"

#include <iostream>
#include "transpose.h"

void transpose(float* input, float* output, int I, int J) {
LOOP_I:
	for (int i = 0; i < I; ++i) {
#pragma HLS loop_tripcount min=10 max=seq_len
		LOOP_J :
		for (int j = 0; j < J; ++j) {
#pragma HLS loop_tripcount min=10 max=d_model
			output[j * I + i] = input[i * J + j]; //IJK->JIK
		}
	}
}

void transpose_12(float* input, float* output, int I, int J, int K) {
LOOP_I:
	for (int i = 0; i < I; ++i) {
#pragma HLS loop_tripcount min=12 max=seq_len
		LOOP_J:
		for (int j = 0; j < J; ++j) {
#pragma HLS loop_tripcount min=12 max=seq_len
			LOOP_K :
			for (int k = 0; k < K; ++k) {
#pragma HLS loop_tripcount max=64
				output[j * I * K + i * K + k] = input[i * J * K + j * K + k]; //IJK->JIK
			}
		}
	}
}

void transpose_23(float* input, float* output, int I, int J, int K) {
LOOP_I:
	for (int i = 0; i < I; ++i) {
#pragma HLS loop_tripcount max=12 
		LOOP_J :
		for (int j = 0; j < J; ++j) {
#pragma HLS loop_tripcount min=seq_len max=64
			LOOP_K :
			for (int k = 0; k < K; ++k) {
#pragma HLS loop_tripcount min=seq_len max=64
				output[i * K * J + k * J + j] = input[i * J * K + j * K + k];//IJK->IKJ
			}
		}
	}
}


