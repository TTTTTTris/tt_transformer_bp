#include "order_control.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <string>
using namespace std;

float input[12 * 8 * 8 * Batchsize * seq_len]{0};
float grad_output[12 * 8 * 8 * Batchsize * seq_len]{0};
struct S0_type { float* x; };
//TYPE_WEIGHT tt_cores_attnq[num_cores * WD_ATTN]{0};
//TYPE_WEIGHT tt_cores_attnk[num_cores * WD_ATTN]{0};
//TYPE_WEIGHT tt_cores_attnv[num_cores * WD_ATTN]{0};
//TYPE_WEIGHT tt_cores_attnfc[num_cores * WD_ATTN]{0};
//TYPE_WEIGHT tt_cores_ff1[num_cores * WD_FFN]{ 0 };
//TYPE_WEIGHT tt_cores_ff2[num_cores * WD_FFN]{ 0 };
TYPE_WEIGHT grad_cores_ff1[num_cores * WD_FFN]{ 0 };
TYPE_WEIGHT grad_cores_ff2[num_cores * WD_FFN]{ 0 };
TYPE_WEIGHT grad_cores_attnq[num_cores * WD_ATTN]{ 0 };
TYPE_WEIGHT grad_cores_attnk[num_cores * WD_ATTN]{ 0 };
TYPE_WEIGHT grad_cores_attnv[num_cores * WD_ATTN]{ 0 };
TYPE_WEIGHT grad_cores_attnfc[num_cores * WD_ATTN]{ 0 };

int getBinSize(const char* file_path) {
	ifstream file(file_path, ios::binary);

	file.seekg(0, ios::end);
	streampos file_size = file.tellg();
	file.seekg(0, ios::beg);

	file.close();
	return file_size;
}

void readBin(float* data, const char* file_path, int file_size) {
	ifstream file(file_path, ios::binary);
	int size = file_size / sizeof(float);
	file.read(reinterpret_cast<char*>(data), size);
	file.close();
}

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

	/* Define FC Layers Cores */
	TYPE_WEIGHT tt_cores_ff1[num_cores * WD_FFN]{ 0 };
	TYPE_WEIGHT tt_cores_ff2[num_cores * WD_FFN]{ 0 };
	TYPE_WEIGHT bias_ff1[12 * 16 * 16]{ 0 };
	TYPE_WEIGHT bias_ff2[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT fc_layernorm_w[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT fc_layernorm_b[12 * 8 * 8]{ 0 };
	/* Define ATTN Layers Cores */
	TYPE_WEIGHT tt_cores_attnq[num_cores * WD_ATTN]{0};
	TYPE_WEIGHT tt_cores_attnk[num_cores * WD_ATTN]{0};
	TYPE_WEIGHT tt_cores_attnv[num_cores * WD_ATTN]{0};
	TYPE_WEIGHT tt_cores_attnfc[num_cores * WD_ATTN]{0};
	TYPE_WEIGHT bias_q[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT bias_k[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT bias_v[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT bias_o[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT attn_layernorm_w[12 * 8 * 8]{ 0 };
	TYPE_WEIGHT attn_layernorm_b[12 * 8 * 8]{ 0 };

	/* Define FC Layers Core Shape */
	 int pff_core_shape[2][6];
	 for (int i = 0; i < 2; i++) {
		 for (int j = 0; j < num_cores; j++) {
			 pff_core_shape[i][j] = tt_ranks[j] * pff_shape[i][j] * tt_ranks[j + 1];
		 }
	 }

	/* Define ATTN Layers Core Shape */
	int attn_core_shape[6];
	for (int j = 0; j < num_cores; j++) {
		attn_core_shape[j] = tt_ranks[j] * attn_shape[j] * tt_ranks[j + 1];
	}

	char file_name[20];
	int file_size{0};
	for (int k = 0; k < num_cores; k++) {
		/* Load FC Layers */
		const string file_fc1 = "ffn/ffn" + to_string(k) + ".txt";
		strcpy(file_name, file_fc1.c_str());
		load_file(tt_cores_ff1 + WD_FFN * k, file_name, pff_core_shape[0][k]); //FC 1
		//file_size = getBinSize(file_name);
		//readBin(tt_cores_ff1 + WD_FFN * k, file_name, file_size);

		const string file_fc2 = "ffn/ffn" + to_string(k + 6) + ".txt";
		strcpy(file_name, file_fc2.c_str());
		load_file(tt_cores_ff2 + WD_FFN * k, file_name, pff_core_shape[1][k]); //FC 2
		//file_size = getBinSize(file_name);
		//readBin(tt_cores_ff2 + WD_FFN * k, file_name, file_size);

		/* Load ATTN Layers */
		const string file_attn1 = "attn/attn" + to_string(k) + ".txt";
		strcpy(file_name, file_attn1.c_str());
		load_file(tt_cores_attnq + WD_ATTN * k, file_name, attn_core_shape[k]); //ATTN_Q
		const string file_attn2 = "attn/attn" + to_string(k + num_cores) + ".txt";
		strcpy(file_name, file_attn2.c_str());
		load_file(tt_cores_attnk + WD_ATTN * k, file_name, attn_core_shape[k]); //ATTN_K
		const string file_attn3 = "attn/attn" + to_string(k + num_cores*2) + ".txt";
		strcpy(file_name, file_attn3.c_str());
		load_file(tt_cores_attnv + WD_ATTN * k, file_name, attn_core_shape[k]); //ATTN_V
		const string file_attn4 = "attn/attn" + to_string(k + num_cores*3) + ".txt";
		strcpy(file_name, file_attn4.c_str());
		load_file(tt_cores_attnfc + WD_ATTN * k, file_name, attn_core_shape[k]); //ATTN_FC
	}

	load_file(input, "attn/input.txt", d_model* Batchsize*seq_len);
	load_file(grad_output, "ffn/FFN_out_grad.txt", d_model * Batchsize*seq_len);
	load_file(bias_q, "attn/bias0.txt", 12 * 8 * 8);
	load_file(bias_k, "attn/bias1.txt", 12 * 8 * 8);
	load_file(bias_v, "attn/bias2.txt", 12 * 8 * 8);
	load_file(bias_o, "attn/bias3.txt", 12 * 8 * 8);
	load_file(bias_ff1, "ffn/bias0.txt", 12 * 16 * 16);
	load_file(bias_ff2, "ffn/bias1.txt", 12 * 8 * 8);
	load_file(attn_layernorm_w, "attn/ln0.txt", 12 * 8 * 8);
	load_file(attn_layernorm_b, "attn/ln1.txt", 12 * 8 * 8);
	load_file(fc_layernorm_w, "ffn/ln0.txt", 12 * 8 * 8);
	load_file(fc_layernorm_b, "ffn/ln1.txt", 12 * 8 * 8);

	order_control_tt_grad_top(
		tt_cores_attnq, tt_cores_attnk, tt_cores_attnv, tt_cores_attnfc, tt_cores_ff1, tt_cores_ff2,
		bias_q, bias_k, bias_v, bias_o, bias_ff1, bias_ff2,
		attn_layernorm_w, attn_layernorm_b, fc_layernorm_w, fc_layernorm_b,
		tt_ranks, attn_shape, pff_shape[0], pff_shape[1]);

	FILE* file1;
	fopen_s(&file1, "grad_cores_q.txt", "w");
	FILE* file2;
	fopen_s(&file2, "grad_cores_k.txt", "w");
	FILE* file3;
	fopen_s(&file3, "grad_cores_v.txt", "w");
	FILE* file4;
	fopen_s(&file4, "grad_cores_fc.txt", "w");
	FILE* file5;
	fopen_s(&file5, "grad_cores_ff1.txt", "w");
	FILE* file6;
	fopen_s(&file6, "grad_cores_ff2.txt", "w");
	for (int i = 0; i < WD_ATTN * num_cores; i++) {
		fprintf(file1, "%f ", grad_cores_attnq[i]);
		fprintf(file2, "%f ", grad_cores_attnk[i]);
		fprintf(file3, "%f ", grad_cores_attnv[i]);
		fprintf(file4, "%f ", grad_cores_attnfc[i]);
	}
	for (int i = 0; i < WD_FFN * num_cores; i++) {
		fprintf(file5, "%f ", grad_cores_ff1[i]);
		fprintf(file6, "%f ", grad_cores_ff2[i]);
	}
	fclose(file1);
	fclose(file2);
	fclose(file3);
	fclose(file4);
	fclose(file5);
	fclose(file6);
}
