#include "order_control.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <string>
using namespace std;

float input[12 * 8 * 8 * Batchsize * seq_len]{ 0 };
float grad_output[12 * 8 * 8 * Batchsize * seq_len]{ 0 };
struct S0_type { float* x; };
TYPE_WEIGHT tt_cores_attnq[num_cores * WD]{ 0 };
TYPE_WEIGHT tt_cores_attnk[num_cores * WD]{ 0 };
TYPE_WEIGHT tt_cores_attnv[num_cores * WD]{ 0 };
TYPE_WEIGHT tt_cores_attnfc[num_cores * WD]{ 0 };
TYPE_WEIGHT grad_cores_attnq[num_cores * WD]{ 0 };
TYPE_WEIGHT grad_cores_attnk[num_cores * WD]{ 0 };
TYPE_WEIGHT grad_cores_attnv[num_cores * WD]{ 0 };
TYPE_WEIGHT grad_cores_attnfc[num_cores * WD]{ 0 };

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

	/* Define ATTN Layers Cores */
	TYPE_WEIGHT tt_cores_attnq[num_cores * WD];
	TYPE_WEIGHT tt_cores_attnk[num_cores * WD];
	TYPE_WEIGHT tt_cores_attnv[num_cores * WD];
	TYPE_WEIGHT tt_cores_attnfc[num_cores * WD];
	TYPE_WEIGHT bias_q[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT bias_k[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT bias_v[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT bias_o[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT layernorm_w[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT layernorm_b[12 * 8 * 8]{ 0 };

	/* Define ATTN Layers Core Shape */
	int attn_core_shape[6];
	for (int j = 0; j < num_cores; j++) {
		attn_core_shape[j] = tt_ranks[j] * attn_shape[j] * tt_ranks[j + 1];
	}

	/* Load ATTN Layers */
	char file_name[20];
	for (int k = 0; k < num_cores; k++) {
		/* Load ATTN Layers */
		const string file_attn1 = "attn/attn" + to_string(k) + ".txt";
		strcpy(file_name, file_attn1.c_str());
		load_file(tt_cores_attnq + WD * k, file_name, attn_core_shape[k]); //ATTN_Q
		const string file_attn2 = "attn/attn" + to_string(k + num_cores) + ".txt";
		strcpy(file_name, file_attn2.c_str());
		load_file(tt_cores_attnk + WD * k, file_name, attn_core_shape[k]); //ATTN_K
		const string file_attn3 = "attn/attn" + to_string(k + num_cores * 2) + ".txt";
		strcpy(file_name, file_attn3.c_str());
		load_file(tt_cores_attnv + WD * k, file_name, attn_core_shape[k]); //ATTN_V
		const string file_attn4 = "attn/attn" + to_string(k + num_cores * 3) + ".txt";
		strcpy(file_name, file_attn4.c_str());
		load_file(tt_cores_attnfc + WD * k, file_name, attn_core_shape[k]); //ATTN_FC
	}
	load_file(input, "attn/input.txt", d_model * Batchsize * seq_len);
	load_file(grad_output, "attn/attn_out_grad.txt", d_model * Batchsize * seq_len);
	load_file(bias_q, "attn/bias0.txt", 12 * 8 * 8);
	load_file(bias_k, "attn/bias1.txt", 12 * 8 * 8);
	load_file(bias_v, "attn/bias2.txt", 12 * 8 * 8);
	load_file(bias_o, "attn/bias3.txt", 12 * 8 * 8);
	load_file(layernorm_w, "attn/ln0.txt", 12 * 8 * 8);
	load_file(layernorm_b, "attn/ln1.txt", 12 * 8 * 8);
	
	/* Train ATTN Layers */
	order_control_tt_grad_attn(
		tt_cores_attnq, tt_cores_attnk, tt_cores_attnv, tt_cores_attnfc,
		bias_q, bias_k, bias_v, bias_o,
		layernorm_w, layernorm_b,
		tt_ranks, attn_shape);
	
	FILE* file1;
	fopen_s(&file1, "grad_cores_q.txt", "w");
	FILE* file2;
	fopen_s(&file2, "grad_cores_k.txt", "w");
	FILE* file3;
	fopen_s(&file3, "grad_cores_v.txt", "w");
	FILE* file4;
	fopen_s(&file4, "grad_cores_fc.txt", "w");
	for (int i = 0; i < WD * num_cores; i++) {
		fprintf(file1, "%f ", grad_cores_attnq[i]);
		fprintf(file2, "%f ", grad_cores_attnk[i]);
		fprintf(file3, "%f ", grad_cores_attnv[i]);
		fprintf(file4, "%f ", grad_cores_attnfc[i]);
	}
	fclose(file1);
	fclose(file2);
	fclose(file3);
	fclose(file4);

}
