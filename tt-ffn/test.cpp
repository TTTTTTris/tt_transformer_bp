#include "order_control.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
using namespace std;

float input[12 * 8 * 8 * Batchsize * seq_len];
float grad_output[12 * 8 * 8 * Batchsize * seq_len];
struct S0_type { float* x; };
float grad_cores_1[num_cores * WD]{ 0 };
float grad_cores_2[num_cores * WD]{ 0 };

void load_file(float* data, const char* filename, int size) {
	ifstream file;
	file.open(filename);
	for (int i = 0; i < size; i++)
	{
		file >> data[i];
	}
	file.close();
}

int main() {
	int tt_ranks[7] = { 1, 10, 10, 10, 10, 10, 1 }; 
	int attn_shape[6] = {12, 8, 8, 8, 8, 12};
	int	pff_shape[2][6] = {{12, 8, 8, 12, 16, 16} ,{16, 16, 12, 8, 8, 12}};

	TYPE_WEIGHT tt_cores_ff1[num_cores * WD]{ 0 };
	TYPE_WEIGHT tt_cores_ff2[num_cores * WD]{ 0 };
	TYPE_WEIGHT bias_ff1[12 * 16 * 16]{ 0 };
	TYPE_WEIGHT bias_ff2[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT layernorm_w[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT layernorm_b[12 * 8 * 8]{ 0 };

	// define tensor core shapes
	int pff_core_shape[2][6];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < num_cores; j++) {
			pff_core_shape[i][j] = tt_ranks[j] * pff_shape[i][j] * tt_ranks[j + 1];
		}
	}

	load_file(input, "paras/FFN_input1.txt", 12 * 8 * 8 * Batchsize * seq_len);
	load_file(grad_output, "paras/FFN_out_grad.txt", 12 * 8 * 8 * Batchsize * seq_len);
	load_file(bias_ff1, "paras/bias0.txt", 12 * 16 * 16);
	load_file(bias_ff2, "paras/bias1.txt", 12 * 8 * 8);
	load_file(layernorm_w, "paras/ln0.txt", 12 * 8 * 8);
	load_file(layernorm_b, "paras/ln1.txt", 12 * 8 * 8);

	char file_name[20];
	for (int k = 0; k < num_cores; k++) {
		const string file_fc1 = "paras/ffn" + to_string(k) + ".txt";
		strcpy_s(file_name, file_fc1.c_str());
		load_file(tt_cores_ff1 + WD * k, file_name, pff_core_shape[0][k]); //FC 1
		const string file_fc2 = "paras/ffn" + to_string(k + 6) + ".txt";
		strcpy_s(file_name, file_fc2.c_str());
		load_file(tt_cores_ff2 + WD * k, file_name, pff_core_shape[1][k]); //FC 2
	}

	// order_control_tt_grad(tt_cores_ff1, tt_ranks, pff_shape[0], input, grad_output, grad_cores_ff1);
	// order_control_tt_grad(tt_cores_ff2, tt_ranks, pff_shape[1], input, grad_output, grad_cores);
	order_control_top(tt_cores_ff1, tt_cores_ff2, bias_ff1, bias_ff2, layernorm_w, layernorm_b, 
		tt_ranks, pff_shape[0], pff_shape[1]);

	FILE* file1;
	fopen_s(&file1, "grad_cores1.txt", "w");
	FILE* file2;
	fopen_s(&file2, "grad_cores2.txt", "w");
	for (int i = 0; i < WD * num_cores; i++) {
		fprintf(file1, "%f ", grad_cores_1[i]);
		fprintf(file2, "%f ", grad_cores_2[i]);
	}
	fclose(file1);
	fclose(file2);
}
