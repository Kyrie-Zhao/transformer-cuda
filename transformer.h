#ifndef _TRANSFORMER_H_
#define _TRANSFORMER_H_


#include <cuda.h>


#include<stdio.h>
#include<stdlib.h>
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include<cuda_fp16.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include<float.h>
#include<vector>
#include<queue>
#include <fstream>
#include <string>
#include<stdlib.h>
#include<stdio.h>


//#include "fp16_conversion.h"

#define SHOW_TIMECOST
#define MAX_TOPK_NUM 300
using namespace::std;

class transformer
{
public:

	
	
	void Initialize(int gpu_id, int n_vs, int n_es, int n_hs, int n_hLayer, int n_aHead, int n_is, int n_bs, int maxseqlen, int n_topk, vector<string> fname);
	//, vector<const char*> fname_kernel, vector<const char*> fname_bias, vector<const char*> fname_norm);
	
	void forward(const int *batch_ids, const int *mask_len, bool use_softmax=false);
	
	
	
	float* get_res_vle() {return h_res_vle;}
	int* get_res_idx() {return h_res_idx;}
	
	~transformer();
	/*void get_res(float *h_res_dis, int *h_res_pos, int *h_len)
	{
		h_res_dis = h_C_res_dis  ;
		h_res_pos = h_C_res_pos  ;
		h_len = h_len_res  ;			
	}*/
private:
	int topk_num;
	int vocab_size;
	int embedding_size; //==hidden_size
	int hidden_size; 
	int num_hidden_layers;
	int num_attention_heads;
	int intermediate_size;//filter_size
	//int max_position_embeddings;
	//int type_vocab_size;
	
	int batch_size;
	int max_seq_len;
	
	//int num_labels;
	
	
	
	cublasHandle_t handle;
	
	//host
	float * h_emb_table;
	//float * h_pos_emb_table;
	//float * h_tokentype_emb_table;	
	
	//float * h_final_output_kernel;
	//float * h_final_output_bias;
	
	
	
	//float * h_mask_vle;
	
	//device
	
	//float * mask_vle_dev;
	
	float * final_res;
	//float * h_final_res;
	
	int * mask_length_dev;
	
	int * input_ids_dev;
	//int * input_token_type_ids_dev;
	
	float * emb_table;
	//float * pos_emb_table;
	//float * tokentype_emb_table;
	
	//float *  emb_LayerNorm_beta;
	//float *  emb_LayerNorm_gamma;	
	
	float * emb_layer_output;
	
	float * transformer_input;
	
	vector<float *> query_kernel;
	//vector<float *> query_bias;
	vector<float *> key_kernel;
	//vector<float *> key_bias;
	vector<float *> value_kernel;
	//vector<float *> value_bias;
	
	vector<float *>  query_layer;
	vector<float *>  key_layer;
	vector<float *>  value_layer;
	float * t_query_layer;
	float * t_key_layer;
	float * t_value_layer;
	
	vector<float *> attention_output_transform_kernel;
	
	
	vector<float *>  context_layer;
	float * t_context_layer;
	
	vector<float *>  attention_scores;
	float * attention_scores_down;
	float * attention_scores_up;
	
	float * t_attention_scores;
	
	vector<float *>  attention_output_kernel;
	vector<float *>  attention_output_bias;

	
	vector<float *>  attention_output;
	
	vector<float *>  output_dense_kernel;
	vector<float *>  output_dense_bias;
	
	vector<float *>  layer_output;	
	
	
	vector<float *>  attention_LayerNorm_beta;
	vector<float *>  attention_LayerNorm_gamma;
	
	vector<float *>  output_LayerNorm_beta;
	vector<float *>  output_LayerNorm_gamma;
	
	float *  pooler_LayerNorm_beta;
	float *  pooler_LayerNorm_gamma;
	
	float * pooler_input;
	
	//float * softmax_bias;
	
	/*
	vector<float *>  intermediate_kernel;
	vector<float *>  intermediate_bias;
	//vector<float *>  intermediate_LayerNorm_beta;
	//vector<float *>  intermediate_LayerNorm_gamma;
	
	vector<float *>  intermediate_output;	
	
	vector<float *>  output_dense_kernel;
	vector<float *>  output_dense_bias;
	vector<float *>  output_dense_LayerNorm_beta;
	vector<float *>  output_dense_LayerNorm_gamma;
	
	vector<float *>  layer_output;	
	*/
	
	//float * pooler_input;
	
	//float * pooler_kernel;
	//float * pooler_bias;
	
	//float* pooler_output;
	
	//float * final_output_kernel;
	//float * final_output_bias;
	
	
	
	
	float * d_C_res_dis  ;
	int *d_C_res_pos  ;


	float * h_res_vle  ;
	int *h_res_idx  ;	

	
	
	bool embedding_lookup_posemb_cal(int *input_ids, const int p_batch_size, const int p_max_len, const int p_embedding_size, float max_timescale, float min_timescale);
	bool embedding_lookup(int *input_ids, const int p_batch_size, const int p_max_len, const int p_vocab_size, const int p_embedding_size);
	bool Embedding_PosEmb_Add(float * pos_emb_table, int *input_ids, const int p_batch_size, const int p_max_len, const int p_vocab_size, const int p_embedding_size);
	bool Embedding_PosEmb_TokenType_Add(float* pos_emb_table, float* tokentype_emb_table, int *input_ids, int *input_token_type_ids, const int p_batch_size, const int p_max_len, const int p_vocab_size, const int p_embedding_size);
	
	void attention_layer(float *from_tensor, float *to_tensor, const int p_num_attention_heads, const int p_size_per_head, const int layer_id, const int from_tensor_row, const int from_tensor_col, const int to_tensor_row, const int to_tensor_col);
	//(float *from_tensor, float *to_tensor, const int num_attention_heads, const int size_per_head, const layer_id);
	
	void layer_norm(float *input, float *output, float *gamma, float *beta, int p_vec_size, int p_batch_size);
	
	
	void dense_gpu(cublasHandle_t h, float *i_A, float *i_B, float * i_bias, float *o_C, int in_vec_size, int out_vec_size, int batch_seq_size, string gelu_flag="none");	
	
	void softmax(float *i_A, float *o_A, int row, int col, int total_col, int mod_size, int *mask_len);
	void softmax_lm(float *i_A, float *o_A, int row, int col, int total_col, int mod_size, int * mask_len);
	void softmax_big(float *i_mat, float* o_mat, int i_nRows, int i_nCols);
	
	void split_mat(float *i_A, float *o_C, int A_row, int A_col, int new_row, int new_col);
	void combine_mat(float *i_A, float *o_C, int A_row, int A_col, int new_row, int new_col);
	
	void transpose(float *matrix, const int row, const int col);
	
	void addBias_Gelu(float *i_mat, float *i_bias, float *o_mat, int row, int col);
	void addBias(float *i_mat, float *i_bias, float *o_mat, int row, int col);
	void addBias_tanh(float *i_mat, float *i_bias, float *o_mat, int row, int col);
	void addBias_norm(float *i_mat, float *i_bias, float *o_mat, int row, int col);
	void addBias_relu(float *i_mat, float *i_bias, float *o_mat, int row, int col);
	
	void MatAdd(float *i_mat_left, float *i_mat_right, float *o_mat, int row, int col);
	
	
	void get_first_token(float *i_mat, float *o_mat, const int p_batch_size, const int p_max_len, const int p_hidden_size);
	void get_any_token(float *i_mat, float *o_mat, const int p_batch_size, const int p_max_len, const int p_hidden_size, int *ith_token);
	
	
	void topk_fun(float* input, int batchsize, int length, int topk, float* output, int* output_pos);

	
	void topk_bucket(float *i_mat, float* o_dis, int *o_vec, int * o_len, int i_nRows, int i_nCols, int tn);
	
	void get_topk_heap(int *idx, float *idx_vle, int *len, int batchsize, int topk, int max_ele, int *res_idx, float *res_vle);

	
	void SetOne(float *io_mat, int row, int col);
	void CheckNAN(float *io_mat, int row, int col);

	void read_data(float* res, int n_rows, int n_cols, string filename)
	{
		FILE *pFile = fopen(filename.c_str(), "rb");	
		fread(res, sizeof(float), n_rows * n_cols, pFile);
		fclose(pFile);
	}




	template <class T>
	void save_data(T* res, int n_rows, int n_cols, char *filename)
	{
		FILE *pFile = fopen(filename, "wb");	
		fwrite(res, sizeof(T), n_rows * n_cols, pFile);
		fclose(pFile);
	}

	template <class T>
	void swap(T &a, T &b)
	{
		a = a + b;  
		b = a - b;  
		a = a - b; 
	}
	
	
	void CPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) 
	{
		int a=1;
		float tmp;
		for(int i = 0; i < nr_rows_A * nr_cols_A; i++)
		{
			tmp = (float)rand()/(float)(RAND_MAX/a);
			A[i] = (tmp); 
		}	
	}

};





#endif
