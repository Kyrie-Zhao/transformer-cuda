#include "transformer.h"

#define TILE_DIM 32
#define BLOCK_ROWS 32
#define REDUCE_SIZE 1024
#define SQRT2DPI 0.79788456080286535587989211986876
#define GPU_THREADS 128
#define NEG_MIN -10000.0f

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

const char* cublasGetErrorString(cublasStatus_t status)
{
     switch(status)
     {
         case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
         case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
         case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
         case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
         case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
         case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
         case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
         case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
     }
     return "unknown error";
}


cublasStatus_t checkCublas(cublasStatus_t result)
{
   if (result != CUBLAS_STATUS_SUCCESS) {
     fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
     assert(result == CUBLAS_STATUS_SUCCESS);
   }
   return result;
}


__device__ inline void combine_topk__stride_pos(float* array, int* pos_array, int pos, int k, int stride, float data)
{
	//printf("bbbbbbbbbbbb   %d  %f\n", pos, data);
    float *prev, *next;
    prev = array + (k-1)*stride;
	int *pos_prev = pos_array + (k-1)*stride, *pos_next;
    if(data < *prev)
        return;
    for(int j=k-2; j>=0; j--)
    {
		next = prev;
        prev = prev - stride;
		pos_next = pos_prev;
		pos_prev = pos_prev - stride;
        if(data > *prev)
		{
            *next = *prev;
			*pos_next = *pos_prev;
		}
        else
		{
            *next = data;
			*(pos_next) = pos;
            return;
        }
    }
    *array = data;
	*pos_array = pos;
	
	
}



__device__ inline void combine_topk_stride(float* left, int* left_pos, float* right, int* right_pos, int k, int stride)
{
    int i;
	//int tp = pos;
    float* current = right;
	int* current_pos = right_pos;
    for(i=0; i<k; i++,current+=stride,current_pos+=stride)
    {
		combine_topk__stride_pos(left, left_pos, *current_pos, k, stride, *current);
		//pos+=stride;
    }
}


__device__ inline void combine_topk_stride_all(float* left, int* letf_pos, float* right, int k, int stride, int pos, int length)
{
    int i;
    float* current = right;
	int tp = pos;
    for(i=0; i<k; i++,current+=stride)
    {
		if(tp<length)
		{
			combine_topk__stride_pos(left, letf_pos, tp, k, stride, *current);
			tp+=stride;
		}
		
    }
}

__global__ void top_k_kernel(float* input, int length, int dpt, int skip_stride, int k, float* output, int* output_pos)
{
    extern __shared__ float shared_buffer[];
    float *myPoint = shared_buffer + threadIdx.x;
	int *myPoint_pos = (int *)(shared_buffer + blockDim.x*k + threadIdx.x);
	int base_idx = blockIdx.x*length;
    int tx = threadIdx.x; //blockIdx.x * blockDim.x + threadIdx.x;
    int ThreadNum = blockDim.x;
    int stride = ThreadNum; 
    int i, pos;

    for(i=0; i<k; i++)
	{
        myPoint[i*stride] = NEG_MIN-i;//input[base_idx + tx + i*stride];
		//myPoint_pos[i*stride] = tx + i*stride;
	}
    for(i=0; i<dpt; i++)
	{
		pos = i*skip_stride + tx;
		combine_topk_stride_all(myPoint, myPoint_pos, input + base_idx + pos, k, stride, pos, length);
	}
    __syncthreads();


    for(i=ThreadNum>>1; i>0; i>>=1) 
	{
        if(tx < i)
        {
			combine_topk_stride(myPoint, myPoint_pos, myPoint+i, myPoint_pos+i, k, stride);
        }
        __syncthreads();
    }


	if(tx<k)
	{
		float *outputPoint = output + blockIdx.x * k + tx;
		float *current = myPoint-tx;
		*outputPoint = *(current + tx*stride);
		
		int *outputPoint_pos = output_pos + blockIdx.x * k + tx;
		int *current_pos = myPoint_pos-tx;
		*outputPoint_pos = *(current_pos + tx*stride);		
	}
    
}


void transformer::topk_fun(float* input, int batchsize, int length, int topk, float* output, int* output_pos)
{
	int threads = GPU_THREADS; 
    int stride = threads * topk;
    int dpt = length / (stride);
	if(length % (stride) != 0)
		dpt++;

    int shared_mem_usage = 2*sizeof(float)*topk*threads;
    top_k_kernel<<<batchsize, threads, shared_mem_usage>>>(input, length, dpt, stride, topk, output, output_pos); 

}





__global__ void Bucket_TopN_kernel(float *y, float *o_res_dis, int *o_res_pos, int *o_res_len, int nCols, int DPT, int tn) 
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	
	__shared__ int bucket[1001];
	__shared__ int stop_pos, save_pos; 
	
	int base_idx = nCols*bx;
	int base_idx_res = tn*bx;
		
	//float max_vle = (float)-1.0e30;
	//int max_vle_idx ;
	int offs = tx ;
	//int offs_pos = tx*tn;
   
	register float temp;
	register int temp_i, temp_pos;

	int  i;
	
	/*for(i=0; i<(10001/REDUCE_SIZE + 1); i++)
	{
		temp_i = i*blockDim.x + tx;
		if(temp_i<10001)
		  bucket[temp_i] = 0;
		
	}*/
	if(tx<1001)
		bucket[tx] = 0;

	for (i = 0; i < DPT; i++, offs += 1024)
	{
		if(offs<nCols)// && i_offs<len)
		{
			temp=y[base_idx+offs] + 1.0f;
			//if(temp>0.0f)
			{
				temp_i = llroundf(temp*500.0f);
				//if(temp_i==1000)
				//	printf("bx==%d tx==%d", bx, tx);
				atomicAdd(bucket+temp_i, 1);
			}
		}
	}
	__syncthreads();
	
	if(tx==0)
	{
		save_pos =0;
		int sum=0;
		for(i=1000; i>=0; i--)
		{
			sum += bucket[i];
			//if(i==1000)
			//	printf("xxxxxxxxxxxxxxx sum==%d\n", sum);
			if(sum<tn && sum >(tn*0.5))
			{
				stop_pos = i;
				o_res_len[bx] = sum;
				break;
			}
			if(sum>=tn)
			{
				stop_pos = i;
				o_res_len[bx] = tn;
				//printf("sum==%d\n", sum);
				break;
			}
		}
	}
	
	__syncthreads();
	offs = tx ;
	
	for (i = 0; i < DPT; i++, offs += 1024)
	{
		if(offs<nCols)// && i_offs<len)
		{
			temp=y[base_idx+offs]+1.0f;
			//if(temp>0.0f)
			{
				temp_i = llroundf(temp*500.0f);
				if(temp_i>=stop_pos)
				{
					temp_pos = atomicAdd(&save_pos, 1);
					if(temp_pos>=tn)
						break;
					o_res_pos[base_idx_res + temp_pos] = offs;		
					o_res_dis[base_idx_res + temp_pos] = temp - 1.0f;					
				//atomicAdd(bucket+temp_i, 1);
				}
			}
		}
	}	
	

}


__global__ void Bucket_TopN_kernel_v2(float *y, float *o_res_dis, int * o_res_pos, int * o_res_len, int nCols, int DPT, int tn) 
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	
	__shared__ int bucket[1001];
	__shared__ int stop_pos, save_pos; 
	
	int base_idx = nCols*bx;
	int base_idx_res = tn*bx;
		
	//float max_vle = (float)-1.0e30;
	//int max_vle_idx ;
	int offs = tx ;
	//int offs_pos = tx*tn;
   
	register float temp;
	register int temp_i, temp_pos;

	int  i;
	
	/*for(i=0; i<(10001/REDUCE_SIZE + 1); i++)
	{
		temp_i = i*blockDim.x + tx;
		if(temp_i<10001)
		  bucket[temp_i] = 0;
		
	}*/
	if(tx<1001)
		bucket[tx] = 0;

	for (i = 0; i < DPT; i++, offs += 1024)
	{
		if(offs<nCols)// && i_offs<len)
		{
			temp=y[base_idx+offs];
			
			//if(temp>0.0f)
			{
				temp_i = llroundf(1000.0f*expf(temp)/(1.05f+expf(temp)));
				atomicAdd(bucket+temp_i, 1);
			}
		}
	}
	__syncthreads();
	
	if(tx==0)
	{
		save_pos =0;
		int sum=0;
		for(i=1000; i>=0; i--)
		{
			sum += bucket[i];
			//if(i==1000)
			//	printf("xxxxxxxxxxxxxxx sum==%d\n", sum);
			if(sum<tn && sum >(tn*0.5))
			{
				stop_pos = i;
				o_res_len[bx] = sum;
				break;
			}
			if(sum>=tn)
			{
				stop_pos = i;
				o_res_len[bx] = tn;
				//printf("sum==%d\n", sum);
				break;
			}
		}
	}
	
	__syncthreads();
	offs = tx ;
	
	for (i = 0; i < DPT; i++, offs += 1024)
	{
		if(offs<nCols)// && i_offs<len)
		{
			temp=expf(y[base_idx+offs])/(1.05f+expf(y[base_idx+offs]));//expf(y[base_idx+offs])/(1.05f+expf(y[base_idx+offs]));//y[base_idx+offs];
			//if(temp>0.0f)
			{
				temp_i = llroundf(1000.0f*temp);
				if(temp_i>=stop_pos)
				{
					temp_pos = atomicAdd(&save_pos, 1);
					if(temp_pos>=tn)
						break;
					o_res_pos[base_idx_res + temp_pos] = offs;	
					o_res_dis[base_idx_res + temp_pos] = temp;
					//printf("%f %d\n", temp, offs);
				//atomicAdd(bucket+temp_i, 1);
				}
			}
		}
	}	
	

}




void transformer::topk_bucket(float *i_mat, float* o_dis, int *o_vec, int * o_len, int i_nRows, int i_nCols, int tn) //(CFtype *y, CFtype * o_res, int nCols, int DPT, int tn) 
{
	int num_thread=1024;
	int dpt = i_nCols / num_thread;
	if(i_nCols % num_thread != 0) 
		dpt++;	
	//if(type<2)
	//	Bucket_TopN_kernel<<<i_nRows, num_thread>>>(i_mat, o_dis, o_vec, o_len, i_nCols, dpt, tn);
	//else
		Bucket_TopN_kernel_v2<<<i_nRows, num_thread>>>(i_mat, o_dis, o_vec, o_len, i_nCols, dpt, tn);
	//getLastCudaError("Error in Calling 'kernel'--output_max");		
}










__global__ void chenknan_kernel(float *io_mat, int row, int col, int dpt)
{
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * col + tx;
	int i, pos = base_idx;

	for(i = 0; i < dpt; i++)
	{
		//pos = base_idx + i*blockDim.x;
		if(tx+i*blockDim.x<col)
		{
		if(!(io_mat[pos] == io_mat[pos]))
			printf("pos==%d\n", pos);
		pos += blockDim.x;
		}
	}

}


void transformer::CheckNAN(float *io_mat, int row, int col)
{
	int num_thread=32;
	int dpt = col / num_thread;
	if(col % num_thread != 0) 
		dpt++;	
	chenknan_kernel<<<row, num_thread>>>(io_mat, row, col, dpt);	
	getLastCudaError("Error in Calling 'kernel'--MatAdd_kernel");
}



__global__ void setone_kernel(float *io_mat, int row, int col, int dpt)
{
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * col + tx;
	int i, pos = base_idx;

	for(i = 0; i < dpt; i++)
	{
		//pos = base_idx + i*blockDim.x;
		io_mat[pos] =  (1.0f);
		pos += blockDim.x;
	}

}


void transformer::SetOne(float *io_mat, int row, int col)
{
	int num_thread=32;
	int dpt = col / num_thread;
	if(col % num_thread != 0) 
		dpt++;	
	setone_kernel<<<row, num_thread>>>(io_mat, row, col, dpt);	
	getLastCudaError("Error in Calling 'kernel'--MatAdd_kernel");
}



__global__ void split_mat_kernel(float *i_A, float *o_C, int A_row, int A_col, int new_row, int new_col, int dpt) //nr==bs  nc==out_vec
{
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;
  int i;
  for(i=0; i<new_row; i++)
    o_C[(bx*dpt + by)*(new_row*new_col) + i*new_col + tx] = i_A[(bx*new_row + i) * A_col + by*new_col + tx];

}


void transformer::split_mat(float *i_A, float *o_C, int A_row, int A_col, int new_row, int new_col)
{
  dim3 grid(A_row/new_row, A_col/new_col);
  dim3 threads(new_col);
  split_mat_kernel<<<grid, threads>>>(i_A, o_C, A_row, A_col, new_row, new_col, A_col/new_col); 
}

__global__ void combine_mat_kernel(float *i_A, float *o_C, int A_row, int A_col, int new_row, int new_col, int dpt) //nr==bs  nc==out_vec
{
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;
  int i;
  for(i=0; i<new_row; i++)
    o_C[(bx*new_row + i) * A_col + by*new_col + tx] = i_A[(bx*dpt + by)*(new_row*new_col) + i*new_col + tx];  
}

void transformer::combine_mat(float *i_A, float *o_C, int A_row, int A_col, int new_row, int new_col)
{
  dim3 grid(A_row/new_row, A_col/new_col);
  dim3 threads(new_col);
  combine_mat_kernel<<<grid, threads>>>(i_A, o_C, A_row, A_col, new_row, new_col, A_col/new_col); 
}

__global__ void transposeCoalesced_kernel(float *odata, float *idata, int width, int height, int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	for (int r=0; r < nreps; r++) 
	{
		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) 
		{
			tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
		}

		__syncthreads();

		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) 
		{
			odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
		}
	}
} 

void transformer::transpose(float *matrix, const int row, const int col)
{
	dim3 grid(row/TILE_DIM, col/TILE_DIM);
	dim3 threads(TILE_DIM, BLOCK_ROWS);
	transposeCoalesced_kernel<<<grid, threads>>>(matrix, matrix, row, col, 1);
	
}

__global__ void addBias_Gelu_kernel(float *i_mat, float *i_bias, float * o_mat,  int row, int col, int copy_reps, int dpt)
{
	int bx = blockIdx.x;   //blockIdx.y * gridDim.x + 
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_idx= bx * blockDim.x + ty;
	

	int i;
	extern __shared__ float bias[];	
	__shared__ float* bias_ptr;// = sum_a + bs;
	bias_ptr = bias;
	int tmp = tx + ty*blockDim.x;
	for(i=0; i<copy_reps; i++, tmp += blockDim.x*blockDim.y)
	{		
		if(tmp<col)
			bias_ptr[tmp] = i_bias[tmp];
	}
	__syncthreads();
		
	if(row_idx<row)
	{	
		int base = row_idx*col;
		tmp = tx;
		float vle;
		for(i=0; i<dpt; i++)
		{			
			vle = i_mat[base + tmp] + bias_ptr[tmp];
			//float tf = __half2float(vle);
			//cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
			vle = 0.5* vle * (1.0 + erff(vle / sqrtf(2.0f)));
			//tf = 0.5f * tf * (1.0f + atanhf(SQRT2DPI * (tf * (1.0f + 0.044715f*tf*tf))));
			o_mat[base+tmp] =  (vle);
			tmp += blockDim.x;
 
		}	
	}
}

__global__ void addBias_kernel(float *i_mat, float *i_bias, float * o_mat,  int row, int col, int copy_reps, int dpt)
{
	int bx = blockIdx.x;   //blockIdx.y * gridDim.x + 
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_idx= bx * blockDim.x + ty;
	

	int i;
	extern __shared__ float bias[];	
	__shared__ float* bias_ptr;// = sum_a + bs;
	bias_ptr = bias;
	int tmp = tx + ty*blockDim.x;
	for(i=0; i<copy_reps; i++, tmp += blockDim.x*blockDim.y)
	{		
		if(tmp<col)
			bias_ptr[tmp] = i_bias[tmp];
	}
	__syncthreads();
		
	if(row_idx<row)
	{	
		int base = row_idx*col;
		tmp = tx;
		//float vle;
		for(i=0; i<dpt; i++)
		{			
			o_mat[base+tmp] = i_mat[base + tmp] + bias_ptr[tmp];
			tmp += blockDim.x;
 
		}	
	}
}



__global__ void addBias_kernel_big(float *i_mat, float *i_bias, float * o_mat,  int row, int col)
{
	int bx = blockIdx.x;   //blockIdx.y * gridDim.x + 
	int tx = threadIdx.x;
	//int ty = threadIdx.y;
	int bias_row_idx= bx*blockDim.x + tx;
	

	int i;
	extern __shared__ float bias[];	
	__shared__ float* bias_ptr;// = sum_a + bs;
	bias_ptr = bias;
	if(bias_row_idx<col)
	{
		bias_ptr[tx] = i_bias[bias_row_idx];
		__syncthreads();
		for(i=0; i<row; i++)
		{
			o_mat[i*col + bias_row_idx] = i_mat[i*col + bias_row_idx] + bias_ptr[tx];
		}
	}
}


void transformer::addBias(float *i_mat, float *i_bias, float *o_mat, int row, int col)
{
	if(col<2048)
	{
		int grid = row/TILE_DIM;
		if(row%TILE_DIM != 0)
			grid++;
		dim3 threads(TILE_DIM, TILE_DIM);
		int dpt = col/TILE_DIM;
		if(col%TILE_DIM != 0)
			dpt++;
		int copy_reps = col/(TILE_DIM*TILE_DIM);
		if(col%(TILE_DIM*TILE_DIM) != 0) 
			copy_reps++;	
		addBias_kernel<<<grid, threads, col*sizeof(float)>>>(i_mat,  i_bias, o_mat, row, col, copy_reps, dpt);
	}
	else
	{
		int num_thread = 256;
		int num_block = col/num_thread;
		if(col%num_thread != 0)
			num_block++;
		addBias_kernel_big<<<num_block, num_thread, num_thread*sizeof(float)>>>(i_mat, i_bias, o_mat,  row, col);
	}
	//reduce_all_kernel<<<(bx, by, bz), num_thread>>>(i_A, o_C, len, total_num, dpt);
	//getLastCudaError("Error in Calling 'kernel'--output_max");	
	//fun_Pearson_Correlation_Coefficient(d_input_vec, d_data_base_vec, d_res, VecSize, SentenceNum, tn);
	
}



void transformer::addBias_Gelu(float *i_mat, float *i_bias, float *o_mat, int row, int col)
{
	//int num_thread=32;
	int grid = row/TILE_DIM;
	if(row%TILE_DIM != 0)
		grid++;
	dim3 threads(TILE_DIM, TILE_DIM);
	int dpt = col/TILE_DIM;
	if(col%TILE_DIM != 0)
		dpt++;
	int copy_reps = col/(TILE_DIM*TILE_DIM);
	if(col%(TILE_DIM*TILE_DIM) != 0) 
		copy_reps++;	
	addBias_Gelu_kernel<<<grid, threads, col*sizeof(float)>>>(i_mat,  i_bias, o_mat, row, col, copy_reps, dpt);
	//reduce_all_kernel<<<(bx, by, bz), num_thread>>>(i_A, o_C, len, total_num, dpt);
	//getLastCudaError("Error in Calling 'kernel'--output_max");	
	//fun_Pearson_Correlation_Coefficient(d_input_vec, d_data_base_vec, d_res, VecSize, SentenceNum, tn);
	
}


__global__ void addBias_tanh_kernel(float *i_mat, float *i_bias, float * o_mat,  int row, int col, int copy_reps, int dpt)
{
	int bx = blockIdx.x;   //blockIdx.y * gridDim.x + 
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_idx= bx * blockDim.x + ty;
	

	int i;
	extern __shared__ float bias[];	
	__shared__ float* bias_ptr;// = sum_a + bs;
	bias_ptr = bias;
	int tmp = tx + ty*blockDim.x;
	for(i=0; i<copy_reps; i++, tmp += blockDim.x*blockDim.y)
	{		
		if(tmp<col)
			bias_ptr[tmp] = i_bias[tmp];
	}
	__syncthreads();
		
	if(row_idx<row)
	{	
		int base = row_idx*col;
		tmp = tx;
		float vle;
		for(i=0; i<dpt; i++)
		{			
			vle = i_mat[base + tmp] + bias_ptr[tmp];
			//float tf = __half2float(vle);
			vle = tanhf( vle);
			o_mat[base+tmp] =  (vle);
			tmp += blockDim.x;
 
		}	
	}
}
void transformer::addBias_tanh(float *i_mat, float *i_bias, float *o_mat, int row, int col)
{
	//int num_thread=32;
	int grid = row/TILE_DIM;
	if(row%TILE_DIM != 0)
		grid++;
	dim3 threads(TILE_DIM, TILE_DIM);
	int dpt = col/TILE_DIM;
	if(col%TILE_DIM != 0)
		dpt++;
	int copy_reps = col/(TILE_DIM*TILE_DIM);
	if(col%(TILE_DIM*TILE_DIM) != 0) 
		copy_reps++;	
	addBias_tanh_kernel<<<grid, threads, col*sizeof(float)>>>(i_mat,  i_bias, o_mat, row, col, copy_reps, dpt);

	
}




__global__ void addBias_relu_kernel(float *i_mat, float *i_bias, float * o_mat,  int row, int col, int copy_reps, int dpt)
{
	int bx = blockIdx.x;   //blockIdx.y * gridDim.x + 
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_idx= bx * blockDim.x + ty;
	

	int i;
	extern __shared__ float bias[];	
	__shared__ float* bias_ptr;// = sum_a + bs;
	bias_ptr = bias;
	int tmp = tx + ty*blockDim.x;
	for(i=0; i<copy_reps; i++, tmp += blockDim.x*blockDim.y)
	{		
		if(tmp<col)
			bias_ptr[tmp] = i_bias[tmp];
	}
	__syncthreads();
		
	if(row_idx<row)
	{	
		int base = row_idx*col;
		tmp = tx;
		float vle;
		for(i=0; i<dpt; i++)
		{			
			vle = i_mat[base + tmp] + bias_ptr[tmp];
			//float tf = __half2float(vle);
			vle = vle>0 ? vle:0.0f;
			o_mat[base+tmp] =  (vle);
			tmp += blockDim.x;
 
		}	
	}
}
void transformer::addBias_relu(float *i_mat, float *i_bias, float *o_mat, int row, int col)
{
	//int num_thread=32;
	int grid = row/TILE_DIM;
	if(row%TILE_DIM != 0)
		grid++;
	dim3 threads(TILE_DIM, TILE_DIM);
	int dpt = col/TILE_DIM;
	if(col%TILE_DIM != 0)
		dpt++;
	int copy_reps = col/(TILE_DIM*TILE_DIM);
	if(col%(TILE_DIM*TILE_DIM) != 0) 
		copy_reps++;	
	addBias_relu_kernel<<<grid, threads, col*sizeof(float)>>>(i_mat,  i_bias, o_mat, row, col, copy_reps, dpt);

	
}




__global__ void addBias_norm_kernel(float *i_mat, float *i_bias, float * o_mat,  int row, int col, int copy_reps, int dpt)
{
	int bx = blockIdx.x;   //blockIdx.y * gridDim.x + 
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_idx= bx * blockDim.x + ty;
	

	int i;
	extern __shared__ float bias[];	
	__shared__ float* bias_ptr;// = sum_a + bs;
	bias_ptr = bias;
	int tmp = tx + ty*blockDim.x;
	for(i=0; i<copy_reps; i++, tmp += blockDim.x*blockDim.y)
	{		
		if(tmp<col)
			bias_ptr[tmp] = i_bias[tmp];
	}
	__syncthreads();
		
	if(row_idx<row)
	{	
		int base = row_idx*col;
		tmp = tx;
		//float vle;
		for(i=0; i<dpt; i++)
		{			
			o_mat[base+tmp] = 1.0f/(1.0f+expf(i_mat[base + tmp] + bias_ptr[tmp]));
			tmp += blockDim.x;
 
		}	
	}
}



__global__ void addBias_norm_kernel_big(float *i_mat, float *i_bias, float * o_mat,  int row, int col)
{
	int bx = blockIdx.x;   //blockIdx.y * gridDim.x + 
	int tx = threadIdx.x;
	//int ty = threadIdx.y;
	int bias_row_idx= bx*blockDim.x + tx;
	

	int i;
	extern __shared__ float bias[];	
	__shared__ float* bias_ptr;// = sum_a + bs;
	bias_ptr = bias;
	if(bias_row_idx<col)
	{
		bias_ptr[tx] = i_bias[bias_row_idx];
		__syncthreads();
		for(i=0; i<row; i++)
		{
			o_mat[i*col + bias_row_idx] = 1.0f/(1.1f+expf(i_mat[i*col + bias_row_idx] + bias_ptr[tx]));
		}
	}
}


void transformer::addBias_norm(float *i_mat, float *i_bias, float *o_mat, int row, int col)
{
	if(col<2048)
	{
		int grid = row/TILE_DIM;
		if(row%TILE_DIM != 0)
			grid++;
		dim3 threads(TILE_DIM, TILE_DIM);
		int dpt = col/TILE_DIM;
		if(col%TILE_DIM != 0)
			dpt++;
		int copy_reps = col/(TILE_DIM*TILE_DIM);
		if(col%(TILE_DIM*TILE_DIM) != 0) 
			copy_reps++;	
		addBias_norm_kernel<<<grid, threads, col*sizeof(float)>>>(i_mat,  i_bias, o_mat, row, col, copy_reps, dpt);
	}
	else
	{
		int num_thread = 256;
		int num_block = col/num_thread;
		if(col%num_thread != 0)
			num_block++;
		addBias_norm_kernel_big<<<num_block, num_thread, num_thread*sizeof(float)>>>(i_mat, i_bias, o_mat,  row, col);
	}
	//reduce_all_kernel<<<(bx, by, bz), num_thread>>>(i_A, o_C, len, total_num, dpt);
	//getLastCudaError("Error in Calling 'kernel'--output_max");	
	//fun_Pearson_Correlation_Coefficient(d_input_vec, d_data_base_vec, d_res, VecSize, SentenceNum, tn);
	
}







__global__ void get_first_token_kernel(float *i_mat, float *o_mat, int p_batch_size, int p_max_len, int p_hidden_size, int dpt)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;	

	if(tx<p_hidden_size)
	{	
		int base_idx_input = bx;
		int base_idx_output = bx * p_hidden_size;
	
		int pos = tx;
		int  j;

		int stride = p_max_len*p_hidden_size;
		for(j = 0; j < dpt; j++)
		{
			
			o_mat[base_idx_output+pos] = i_mat[bx * stride  + pos];
			pos += blockDim.x;
		}	
	
	}	
}

void transformer::get_first_token(float *i_mat, float *o_mat, const int p_batch_size, const int p_max_len, const int p_hidden_size)
{
	if(p_hidden_size>1024)
	{
		printf("ERROR----Hidden size must less than 1024\n");
		exit(0);;
	}
	int num_thread = 32;
	int dpt = p_hidden_size/num_thread;
	get_first_token_kernel<<<p_batch_size, num_thread>>>(i_mat, o_mat, p_batch_size, p_max_len, p_hidden_size, dpt);
	//return true;	
}




__global__ void get_any_token_kernel(float *i_mat, float *o_mat, int p_batch_size, int p_max_len, int p_hidden_size, int *ith_token, int dpt)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;	

	if(tx<p_hidden_size)
	{	
		int base_idx_input = bx;
		int base_idx_output = bx * p_hidden_size;
	
		int pos = tx;
		int  j;
		int start_pos = (ith_token[bx]-1) * p_hidden_size;

		int stride = p_max_len*p_hidden_size;
		for(j = 0; j < dpt; j++)
		{
			
			o_mat[base_idx_output+pos] = i_mat[bx * stride + start_pos + pos];
			pos += blockDim.x;
		}	
	
	}	
}

void transformer::get_any_token(float *i_mat, float *o_mat, const int p_batch_size, const int p_max_len, const int p_hidden_size, int *ith_token)
{
	if(p_hidden_size>1024)
	{
		printf("ERROR----Hidden size must less than 1024\n");
		exit(0);;
	}
	int num_thread = 32;
	int dpt = p_hidden_size/num_thread;
	get_any_token_kernel<<<p_batch_size, num_thread>>>(i_mat, o_mat, p_batch_size, p_max_len, p_hidden_size, ith_token, dpt);
	//return true;	
}




/*__global__ void embbeding_lookup_pos_cal_kernel(float *embed, int *input_ids, float *o_mat, int emb_size, int max_len, float min_timescale, float log_timescale_increment, int dpt)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;	

	if(tx<emb_size)
	{	
		int base_idx_input = bx;
		int base_idx_output = bx * emb_size;
	
		int pos_s = tx, pos_c = tx + emb_size/2;
		int  j;


		for(j = 0; j < dpt; j++)
		{
			
			o_mat[base_idx_output+pos_s] = sqrtf(emb_size)*embed[input_ids[base_idx_input] * emb_size + pos_s] + sinf((base_idx_input%max_len) * min_timescale * expf(pos_s * -log_timescale_increment));
			o_mat[base_idx_output+pos_c] = sqrtf(emb_size)*embed[input_ids[base_idx_input] * emb_size + pos_c] + cosf((base_idx_input%max_len) * min_timescale * expf(pos_s * -log_timescale_increment));
			
			pos_s += blockDim.x;
			pos_c += blockDim.x;
		}	
	
	}
}

bool transformer::embedding_lookup_posemb_cal(int *input_ids, const int p_batch_size, const int p_max_len, const int p_embedding_size, float max_timescale, float min_timescale )
{
	if(p_embedding_size>1024)
	{
		printf("ERROR----Embbeding size must less than 1024\n");
		return false;
	}
	int num_thread = 32;
	int dpt = p_embedding_size/(2*num_thread);
	float log_timescale_increment = ( log((max_timescale) / (min_timescale)) / (float(p_embedding_size/2) - 1.0));
	embbeding_lookup_pos_cal_kernel<<<p_batch_size*p_max_len, num_thread>>>(emb_table, input_ids, emb_layer_output, p_embedding_size, p_max_len, min_timescale, log_timescale_increment, dpt);
	return true;
}	*/

__global__ void embbeding_lookup_pos_cal_kernel(float *embed, int *input_ids, float *o_mat, int emb_size, int max_len, float min_timescale, float log_timescale_increment, int dpt)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;	

	if(tx<emb_size)
	{	
		int base_idx_input = bx;
		int base_idx_output = bx * emb_size;
	
		int pos_s = tx, pos_c = tx + emb_size/2;
		int  j;

		for(j = 0; j < dpt; j++)
		{
			
			o_mat[base_idx_output+pos_s] =  sqrtf(emb_size)*embed[input_ids[base_idx_input] * emb_size + pos_s] + sinf((base_idx_input%max_len) * min_timescale * expf(pos_s * -log_timescale_increment));
			//pos_s += blockDim.x;
			o_mat[base_idx_output+pos_c] =  sqrtf(emb_size)*embed[input_ids[base_idx_input] * emb_size + pos_c] + cosf((base_idx_input%max_len) * min_timescale * expf(pos_s * -log_timescale_increment));
			pos_c += blockDim.x;
			pos_s += blockDim.x;
		}	
	
	}
}

bool transformer::embedding_lookup_posemb_cal(int *input_ids, const int p_batch_size, const int p_max_len, const int p_embedding_size, float min_timescale, float max_timescale )
{
	if(p_embedding_size>1024)
	{
		printf("ERROR----Embbeding size must less than 1024\n");
		return false;
	}
	int num_thread = 32;
	int dpt = p_embedding_size/(2*num_thread);
	float log_timescale_increment = ( log((max_timescale) / (min_timescale)) / (float(p_embedding_size/2) - 1.0));
	embbeding_lookup_pos_cal_kernel<<<p_batch_size*p_max_len, num_thread>>>(emb_table, input_ids, emb_layer_output, p_embedding_size, p_max_len, min_timescale, log_timescale_increment, dpt);
	return true;
}





__global__ void embbeding_lookup_kernel(float *embed, int *input_ids, float *o_mat, int batch_size, int emb_size, int dpt)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;	

	if(tx<emb_size)
	{	
		int base_idx_input = bx;
		int base_idx_output = bx * emb_size;
	
		int pos = tx;
		int  j;


		for(j = 0; j < dpt; j++)
		{
			
			o_mat[base_idx_output+pos] = embed[input_ids[base_idx_input] * emb_size + pos];
			pos += blockDim.x;
		}	
	
	}
}

bool transformer::embedding_lookup(int *input_ids, const int p_batch_size, const int p_max_len, const int p_vocab_size, const int p_embedding_size)
{
	if(p_embedding_size>1024)
	{
		printf("ERROR----Embbeding size must less than 1024\n");
		return false;
	}
	int num_thread = 32;
	int dpt = p_embedding_size/num_thread;
	embbeding_lookup_kernel<<<p_batch_size*p_max_len, num_thread>>>(emb_table, input_ids, emb_layer_output, p_batch_size, p_embedding_size, dpt);
	return true;
}

__global__ void Embedding_PosEmb_Add_kernel(float *embed, float *pos_emb, int *input_ids, float *o_mat, int batch_size, int emb_size, int max_len, int dpt)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;	

	if(tx<emb_size)
	{	
		int base_idx_input = bx;
		int base_idx_output = bx * emb_size;
	
		int pos = tx;
		int  j;


		for(j = 0; j < dpt; j++)
		{
			
			o_mat[base_idx_output+pos] = embed[input_ids[base_idx_input] * emb_size + pos] + pos_emb[(base_idx_input%max_len) * emb_size + pos];
			pos += blockDim.x;
		}	
	
	}
}


bool transformer::Embedding_PosEmb_Add(float * pos_emb_table, int *input_ids, const int p_batch_size, const int p_max_len, const int p_vocab_size, const int p_embedding_size)
{
	if(p_embedding_size>1024)
	{
		printf("ERROR----Embbeding size must less than 1024\n");
		return false;
	}
	int num_thread = 32;
	int dpt = p_embedding_size/num_thread;
	Embedding_PosEmb_Add_kernel<<<p_batch_size*p_max_len, num_thread>>>(emb_table, pos_emb_table, input_ids, emb_layer_output, p_batch_size, p_embedding_size, p_max_len, dpt);
	return true;	
}




__global__ void Embedding_PosEmb_TokenType_Add_kernel(float *embed, float *pos_emb, float *tokentype_emb, int *input_ids, int *input_token_type_ids, float *o_mat, int batch_size, int emb_size, int max_len, int dpt)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;	

	if(tx<emb_size)
	{	
		int base_idx_input = bx;
		int base_idx_output = bx * emb_size;
	
		int pos = tx;
		int  j;


		for(j = 0; j < dpt; j++)
		{
			
			o_mat[base_idx_output+pos] = embed[input_ids[base_idx_input] * emb_size + pos] + tokentype_emb[input_token_type_ids[base_idx_input] * emb_size + pos] + pos_emb[(base_idx_input%max_len) * emb_size + pos];
			pos += blockDim.x;
		}	
	
	}
}

bool transformer::Embedding_PosEmb_TokenType_Add(float* pos_emb_table, float* tokentype_emb_table, int *input_ids, int *input_token_type_ids, const int p_batch_size, const int p_max_len, const int p_vocab_size, const int p_embedding_size)
{
	if(p_embedding_size>1024)
	{
		printf("ERROR----Embbeding size must less than 1024\n");
		return false;
	}
	int num_thread = 32;
	int dpt = p_embedding_size/num_thread;
	Embedding_PosEmb_TokenType_Add_kernel<<<p_batch_size*p_max_len, num_thread>>>(emb_table, pos_emb_table, tokentype_emb_table, input_ids, input_token_type_ids, emb_layer_output, p_batch_size, p_embedding_size, p_max_len, dpt);
	return true;		
}



__global__ void layer_norm_kernel(float *i_A, float *o_C, float *i_gamma, float *i_beta, int vec_len, int bs, int dpt)
{
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * vec_len + tx;
	float sum_aa = 0.0f;
	int i, pos;
	__shared__ float mean, var;
	
	//cal mean
	for(i = 0; i < dpt; i++)
	{
		pos = base_idx + i*blockDim.x;		
		sum_aa += i_A[pos];				
	}
	
	for (int i = 16; i >= 1; i /= 2)
	{
//#ifndef SMFOR35
//		sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
//#else
		sum_aa += __shfl_xor(sum_aa, i, 32);
//#endif
	}
	
	//for (int i=16; i>0; i=i/2)
	//	sum_aa += __shfl_down_sync(0xffffffff, sum_aa, i);	

	if(tx==0)
	{
		//mean = __hdiv(sum_aa, __int2half_rn (vec_len));
		mean = sum_aa / float(vec_len);
	}
	__syncthreads();
	
	
	//cal var
	sum_aa = 0.0;
	float tmp;

	for(i = 0; i < dpt; i++)
	{
		pos = base_idx + i*blockDim.x;	
		tmp	= (i_A[pos] - mean);
		sum_aa += tmp*tmp;				
	}
	for (int i = 16; i >= 1; i /= 2)
	{
//#ifndef SMFOR35
//		sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
//#else
		sum_aa += __shfl_xor(sum_aa, i, 32);
//#endif
	}
	
	//for (int i=16; i>0; i=i/2)
	//	sum_aa += __shfl_down_sync(0xffffffff, sum_aa, i);	

	if(tx==0)
	{
		//var = __hdiv(sum_aa, __int2half_rn (vec_len));
		var = sum_aa / float(vec_len);
	}
	__syncthreads();	

	
	//cal norm 
	for(i = 0; i < dpt; i++)
	{
		pos = base_idx + i*blockDim.x;	
		o_C[pos] = i_gamma[tx + i*blockDim.x] * ((i_A[pos] - mean) / sqrtf(var+ (0.0000001f))) + i_beta[tx + i*blockDim.x];
	}		
}

void transformer::layer_norm(float *input, float *output, float *gamma, float *beta, int p_vec_size, int p_batch_size)
{
	int num_thread=32;
	int dpt = p_vec_size / num_thread;
	if(p_vec_size % num_thread != 0) 
		dpt++;	
	layer_norm_kernel<<<p_batch_size, num_thread>>>(input, output, gamma, beta, p_vec_size, p_batch_size, dpt);
	getLastCudaError("Error in Calling 'kernel'--layer_norm_kernel");
}


/*
__global__ void layer_norm_kernel(float *i_A, float *o_C, float *i_gamma, float *i_beta, int vec_len, int bs, int dpt)
{
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * vec_len + tx;
	float sum_aa = 0.0;
	int i, pos;
	__shared__ float mean, var;
	
	//cal mean
	for(i = 0; i < dpt; i++)
	{
		pos = base_idx + i*blockDim.x;		
		sum_aa += i_A[pos];				
	}
	for (int i=16; i>0; i=i/2)
		sum_aa += __shfl_down_sync(0xffffffff, sum_aa, i);	
	//for (int i = 16; i >= 1; i /= 2)
	//{
		//sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
	//	sum_aa += __shfl_xor (sum_aa, i, 32);
	//}
	if(tx==0)
	{
		mean = __hdiv(sum_aa, __int2half_rn (vec_len));
	}
	__syncthreads();
	
	
	//cal var
	sum_aa = 0.0;
	float tmp;

	for(i = 0; i < dpt; i++)
	{
		pos = base_idx + i*blockDim.x;	
		tmp	= (i_A[pos] - mean);
		sum_aa += tmp*tmp;				
	}
	for (int i=16; i>0; i=i/2)
		sum_aa += __shfl_down_sync(0xffffffff, sum_aa, i);	

	if(tx==0)
	{
		var = __hdiv(sum_aa, __int2half_rn (vec_len));
	}
	__syncthreads();	

	
	//cal norm 
	for(i = 0; i < dpt; i++)
	{
		pos = base_idx + i*blockDim.x;	
		o_C[pos] = i_gamma[bx] * ((i_A[pos] - mean) / hrsqrt(var+ (0.0000001f))) + i_beta[bx];
	}		
}



void transformer::layer_norm(float *input, float *output, float *gamma, float *beta, int p_vec_size, int p_batch_size)
{
	transpose(input, p_batch_size, p_vec_size);
	int num_thread=32;
	int dpt = p_batch_size / num_thread;
	if(p_batch_size % num_thread != 0) 
		dpt++;	
	layer_norm_kernel<<<p_vec_size, num_thread>>>(input, output, gamma, beta, p_batch_size, p_vec_size, dpt);
	getLastCudaError("Error in Calling 'kernel'--layer_norm_kernel");
	transpose(output, p_vec_size, p_batch_size);
}*/


__global__ void softmax_kernel(float *i_A, float *o_C, int n_row, int n_col, int total_col, int mod_size, int * mask_len, int dpt) 
{
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * n_col;
	int row_idx= tx;
	//if(base_idx<total_col)
	{
		__shared__ float sum;
		float sum_aa = (0.0f);
		int i, pos;	
		for(i=0; i<dpt; i++)
		{
			pos = row_idx+i*blockDim.x;
			if(pos<mask_len[bx/mod_size])
				sum_aa += expf(i_A[base_idx+pos]);
		}
		for (int i = 16; i >= 1; i /= 2)
		{
//#ifndef SMFOR35
//			sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
//#else
			sum_aa += __shfl_xor(sum_aa, i, 32);
//#endif
		}
		//for (i=16; i>0; i=i/2)
		//	sum_aa += __shfl_down_sync(0xffffffff, sum_aa, i);
		__syncthreads();
		if(tx==0)
		{
			sum = sum_aa;
		}
		__syncthreads();
		for(i=0; i<dpt; i++)
		{
			pos = row_idx+i*blockDim.x;
			if(pos<mask_len[bx/mod_size])
				//o_C[base_idx+pos] = __hdiv(hexp(i_A[base_idx+pos]), sum);
				o_C[base_idx+pos] = expf(i_A[base_idx+pos]) / sum;
			else
				o_C[base_idx+pos] = (0.0f);
		}		
	}
	/*int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * blockDim.x + tx;
	if(base_idx<total_col)
	{
		float sum_aa = 0.0;
		int i;	
		for(i=0; i<mask_len[base_idx]; i++)
		{
			sum_aa += hexp(i_A[i + n_col*base_idx]);
		}
		for(i=0; i<mask_len[base_idx]; i++)
		{
			o_C[i + n_col*base_idx] = __hdiv(hexp(i_A[i + n_col*base_idx]) , sum_aa);
		}		
		for(i=mask_len[base_idx]; i<n_row; i++)
		{
			o_C[i + n_col*base_idx] = (0.0f);
		}		
	}*/
	
}

__global__ void softmax_kernel_v2(float *i_A, float *o_C, int n_row, int n_col, int total_col, int mod_size, int * mask_len) 
{
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * n_col;
	int row_idx= tx;
	//if(base_idx<total_col)
	{
		float sum_aa = 0.0f;
		int i, pos;	
		for(i=0; i<mask_len[bx/mod_size]; i++)
		{
			sum_aa += expf(i_A[base_idx+i]);
		}
		for(i=0; i<mask_len[bx/mod_size]; i++)
			//o_C[base_idx+i] = __hdiv(hexp(i_A[base_idx+i]), sum_aa);
			o_C[base_idx+i] = expf(i_A[base_idx+i]) / sum_aa;
		for(i=mask_len[bx/mod_size]; i<n_col; i++)
			o_C[base_idx+i] = (0.0f);
	}	
}


void transformer::softmax(float *i_A, float *o_A, int row, int col, int total_col, int mod_size, int * mask_len)
{
	/*int num_thread=1024;
	int dpt = total_col / num_thread;
	if(total_col % num_thread != 0) 
		dpt++;	
	softmax_kernel<<<dpt, num_thread>>>(i_A, o_A, row, col, total_col, mod_size, mask_len, dpt);
	getLastCudaError("Error in Calling 'kernel'--softmax_kernel");	
	*/
	int num_thread=32;
	int dpt = col / num_thread;
	if(col % num_thread != 0) 
		dpt++;	
	if(dpt!=1)
		softmax_kernel<<<total_col, num_thread>>>(i_A, o_A, row, col, total_col, mod_size, mask_len, dpt);
	else
		softmax_kernel_v2<<<total_col, 1>>>(i_A, o_A, row, col, total_col, mod_size, mask_len);
	getLastCudaError("Error in Calling 'kernel'--softmax_kernel");
	
}




__global__ void softmax_lm_kernel(float *i_A, float *o_C, int n_row, int n_col, int total_col, int mod_size, int * mask_len, int dpt) 
{
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * n_col;
	int row_idx= tx;
	//int bias_idx = bx%mask_len[bx/mod_size]; 
	//int bias_idx = (bx%mod_size)%mask_len[bx/mod_size]; 
	int bias_idx = (bx%n_col)%mask_len[bx/mod_size]; 
	//if(base_idx<total_col)
	{
		__shared__ float sum;
		float sum_aa = (0.0f);
		int i, pos;	
		for(i=0; i<dpt; i++)
		{
			pos = row_idx+i*blockDim.x;
			//if(pos<mask_len[bx/mod_size])
			if(pos<=bias_idx)
				sum_aa += expf(i_A[base_idx+pos]);
			//if(pos>bias_idx && pos<mask_len[bx/mod_size])
				
		}
		for (int i = 16; i >= 1; i /= 2)
		{
//#ifndef SMFOR35
//			sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
//#else
			sum_aa += __shfl_xor(sum_aa, i, 32);
//#endif
		}
		//for (i=16; i>0; i=i/2)
		//	sum_aa += __shfl_down_sync(0xffffffff, sum_aa, i);
		__syncthreads();
		if(tx==0)
		{
			sum = sum_aa;
		}
		__syncthreads();
		for(i=0; i<dpt; i++)
		{
			pos = row_idx+i*blockDim.x;
			//if(pos<mask_len[bx/mod_size])
			if(pos<=bias_idx)
				//o_C[base_idx+pos] = __hdiv(hexp(i_A[base_idx+pos]), sum);
				o_C[base_idx+pos] = expf(i_A[base_idx+pos]) / sum;
			else
				o_C[base_idx+pos] = (0.0f);
		}		
	}
	
}



void transformer::softmax_lm(float *i_A, float *o_A, int row, int col, int total_col, int mod_size, int * mask_len)
{
	int num_thread=32;
	int dpt = col / num_thread;
	if(col % num_thread != 0) 
		dpt++;	

	softmax_lm_kernel<<<total_col, num_thread>>>(i_A, o_A, row, col, total_col, mod_size, mask_len, dpt);

	getLastCudaError("Error in Calling 'kernel'--softmax_kernel");
	
}




__global__ void MatAdd_kernel(float *i_mat_left, float *i_mat_right, float *o_mat, int row, int col, int dpt)
{
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * col + tx;
	int i, pos = base_idx;

	for(i = 0; i < dpt; i++)
	{
		//pos = base_idx + i*blockDim.x;
		o_mat[pos] = i_mat_left[pos] + i_mat_right[pos];
		pos += blockDim.x;
	}

}


void transformer::MatAdd(float *i_mat_left, float *i_mat_right, float *o_mat, int row, int col)
{
	int num_thread=32;
	int dpt = col / num_thread;
	if(col % num_thread != 0) 
		dpt++;	
	MatAdd_kernel<<<row, num_thread>>>(i_mat_left, i_mat_right, o_mat, row, col, dpt);	
	getLastCudaError("Error in Calling 'kernel'--MatAdd_kernel");
}


void transformer::dense_gpu(cublasHandle_t h, float *i_A, float *i_B, float * i_bias, float *o_C, int in_vec_size, int out_vec_size, int batch_seq_size, string gelu_flag)
{


    const float alf = (1.0f);
    const float bet = (0.0f);
    const float *alpha = &alf;
    const float *beta = &bet;
	//cublasStatus_t stat = cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out_vec_size, batch_seq_size, in_vec_size, alpha, i_B, in_vec_size, i_A, in_vec_size, beta, o_C, out_vec_size);
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, *B, n, *A, k, &beta, *C, n);
	cublasStatus_t stat = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, out_vec_size, batch_seq_size, in_vec_size, alpha, i_B, out_vec_size, i_A, in_vec_size, beta, o_C, out_vec_size);
		
	checkCublas(stat);
	if(gelu_flag=="no_bias")
		return;

	if(gelu_flag=="tanh")
		addBias_tanh(o_C, i_bias, o_C, batch_seq_size, out_vec_size);
	
	if(gelu_flag=="gelu")
		addBias_Gelu(o_C, i_bias, o_C, batch_seq_size, out_vec_size);
	if(gelu_flag=="none")
		addBias(o_C, i_bias, o_C, batch_seq_size, out_vec_size);
	if(gelu_flag=="norm")
		addBias_norm(o_C, i_bias, o_C, batch_seq_size, out_vec_size);
	if(gelu_flag=="relu")
		addBias_relu(o_C, i_bias, o_C, batch_seq_size, out_vec_size);
		
}



void transformer::attention_layer(float *from_tensor, float *to_tensor, const int p_num_attention_heads, const int p_size_per_head, const int layer_id, const int from_tensor_row, const int from_tensor_col, const int to_tensor_row, const int to_tensor_col)
{
	int out_size = p_num_attention_heads*p_size_per_head;
	
	float * bias_none=NULL;
	
	layer_norm(from_tensor, attention_scores_up, attention_LayerNorm_gamma[layer_id], attention_LayerNorm_beta[layer_id], hidden_size, batch_size*max_seq_len);	
	
	


	
	dense_gpu(handle, attention_scores_up, query_kernel[layer_id], bias_none, query_layer[layer_id], from_tensor_col, out_size, from_tensor_row, "no_bias");



	
	dense_gpu(handle, attention_scores_up, key_kernel[layer_id], bias_none, key_layer[layer_id], to_tensor_col, out_size, to_tensor_row, "no_bias");
	
	

	
	
	dense_gpu(handle, attention_scores_up, value_kernel[layer_id], bias_none, value_layer[layer_id], to_tensor_col, out_size, to_tensor_row, "no_bias");
	
	

	
	
	split_mat(query_layer[layer_id], t_query_layer, from_tensor_row, out_size, max_seq_len, p_size_per_head);
	split_mat(key_layer[layer_id], t_key_layer, to_tensor_row, out_size, max_seq_len, p_size_per_head);

	
	//SetOne(t_query_layer, from_tensor_row, out_size);
	//SetOne(t_key_layer, from_tensor_row, out_size);




	float alpha = (1.0f / sqrt(float(p_size_per_head))); 
	float beta = (0.0f);
	cublasStatus_t cublasstat = cublasSgemmStridedBatched(handle,
			CUBLAS_OP_T, 
			CUBLAS_OP_N,
			max_seq_len, max_seq_len, p_size_per_head,
			&alpha,
			(const float*)t_key_layer, p_size_per_head,
			p_size_per_head*max_seq_len,
			(const float*)t_query_layer, p_size_per_head,
			p_size_per_head*max_seq_len,
			&beta,
			t_attention_scores, max_seq_len, 
			max_seq_len*max_seq_len, 
			batch_size*p_num_attention_heads);
			

			
			

	/*std::cout<<std::endl<<"down----layer-"<<layer_id<<endl;
	float * tttres1 = (float *)malloc(batch_size*num_attention_heads*max_seq_len*max_seq_len*(sizeof(float)));
	cudaMemcpy(tttres1, t_attention_scores, batch_size*num_attention_heads*max_seq_len*max_seq_len*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<10; j++)
		std::cout<<(tttres1[i*num_attention_heads*max_seq_len*max_seq_len +j ])<<"  ";
	cout<<endl;
	}	*/			

	

	
			
	
	//softmax(t_attention_scores, t_attention_scores, max_seq_len, max_seq_len, batch_size*p_num_attention_heads*max_seq_len, p_num_attention_heads*max_seq_len, mask_length_dev);

	softmax_lm(t_attention_scores, t_attention_scores, max_seq_len, max_seq_len, batch_size*p_num_attention_heads*max_seq_len, p_num_attention_heads*max_seq_len, mask_length_dev);
	
	
	/*if(layer_id==0)
	{
		std::cout<<std::endl<<"up----layer-"<<layer_id<<endl;
	float * ttres1 = (float *)malloc(batch_size*num_attention_heads*max_seq_len*max_seq_len*(sizeof(float)));
	cudaMemcpy(ttres1, t_attention_scores, batch_size*num_attention_heads*max_seq_len*max_seq_len*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<4; i++)
	{
		for(int j=0; j<4; j++)
		{
			std::cout<<(ttres1[i*max_seq_len +j])<<"  ";
			//if(ttres1[i*num_attention_heads*max_seq_len*max_seq_len + j]!= ttres1[j])
			//	std::cout<<i<<" "<<j<<" "<<(ttres1[i*num_attention_heads*max_seq_len*max_seq_len +j])<<"  "<<ttres1[j]<<endl;
		}
		cout<<endl;

	}
	}*/


	
	
	
	split_mat(value_layer[layer_id], t_value_layer, to_tensor_row, out_size, max_seq_len, p_size_per_head);




	
	alpha = (1.0f); 
	beta = (0.0f);			
	cublasstat = cublasSgemmStridedBatched(handle,
			CUBLAS_OP_N, 
			CUBLAS_OP_N,
			p_size_per_head, max_seq_len, max_seq_len, 
			&alpha,
			(const float*) t_value_layer, p_size_per_head,
			p_size_per_head*max_seq_len,
			(const float*) t_attention_scores, max_seq_len,
			max_seq_len*max_seq_len,
			&beta,
			t_context_layer, p_size_per_head, 
			max_seq_len*p_size_per_head, 
			batch_size*p_num_attention_heads);	
			


		
			

			
	combine_mat(t_context_layer, context_layer[layer_id], to_tensor_row, out_size, max_seq_len, p_size_per_head);

	


	
	//combine_mat(t_context_layer, t_context_layer, to_tensor_row, out_size, max_seq_len, p_size_per_head);	

	

	
	

	
	
	
	dense_gpu(handle, context_layer[layer_id], attention_output_transform_kernel[layer_id], bias_none, attention_scores[layer_id], hidden_size, hidden_size, batch_size*max_seq_len, "no_bias");


	

	MatAdd(attention_scores[layer_id], from_tensor, attention_scores[layer_id], batch_size*max_seq_len, hidden_size);
	


	
	
	layer_norm(attention_scores[layer_id], attention_scores_down, output_LayerNorm_gamma[layer_id], output_LayerNorm_beta[layer_id], hidden_size, batch_size*max_seq_len);
	


	
	dense_gpu(handle, attention_scores_down, attention_output_kernel[layer_id], attention_output_bias[layer_id], attention_output[layer_id], hidden_size, intermediate_size, batch_size*max_seq_len, "relu");	

		
	
	dense_gpu(handle, attention_output[layer_id], output_dense_kernel[layer_id], output_dense_bias[layer_id], layer_output[layer_id], intermediate_size, hidden_size, batch_size*max_seq_len, "none");	



	
	MatAdd(layer_output[layer_id], attention_scores[layer_id], layer_output[layer_id], batch_size*max_seq_len, hidden_size);

	
/*if(layer_id==3)
{
	float * ttres1 = (float *)malloc(embedding_size*max_seq_len*batch_size*(sizeof(float)));
	cudaMemcpy(ttres1, layer_output[layer_id], embedding_size*batch_size*max_seq_len*sizeof(float), cudaMemcpyDeviceToHost);
	
	cout<<endl;
	for(int a1=0; a1<4; a1++)
	{
		for(int a3=0; a3<10; a3++)
		{
			cout<<ttres1[a1*embedding_size + a3]<<" ";
		}
		cout<<endl;
	}	
}*/
}


//gpu_id== gpu_id  n_vs==vocabsize  n_es==embbeding_size  n_hs=hidden_size  n_hLayer==num_attention_layers  n_aHead==num_heads
//n_is==num_filter_size  n_bs==batch_size 
//maxseqlen==max_seq_len  ttk==topk_num
void transformer::Initialize(int gpu_id, int n_vs, int n_es, int n_hs, int n_hLayer, int n_aHead, int n_is, int n_bs, int maxseqlen, int n_topk, vector<string> fname)
{
	int i;
	cudaError_t err = cudaSuccess;
	
	cudaSetDevice(gpu_id);	

	cublasStatus_t cublasStat = cublasCreate(&handle);
	//cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	
	
	topk_num =  n_topk;
	vocab_size = n_vs;
	embedding_size = n_es;
	hidden_size = n_hs; 
	num_hidden_layers = n_hLayer;
	num_attention_heads = n_aHead;
	intermediate_size = n_is;
	//max_position_embeddings = mpe;
	//type_vocab_size = n_tvs;
	
	batch_size = n_bs;
	max_seq_len = maxseqlen;
	
	//num_labels = n_labels;
	
	
	printf("Allocating memory on host and device...\n");
	
	
	
	
    err = cudaMalloc((void **)&mask_length_dev, batch_size*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mask_length_dev (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	int size_ids = batch_size*max_seq_len*sizeof(int);
    err = cudaMalloc((void **)&input_ids_dev, size_ids);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input_ids_dev (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	
	
	
	//read_data(h_emb_table, vocab_size, embedding_size, fname[0]);
	
	
	
	//emb_table
	int size_emb_table = vocab_size * embedding_size *  sizeof(float);
	h_emb_table = (float *)malloc(size_emb_table);
    if (h_emb_table == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors h_emb_table!\n");
        exit(EXIT_FAILURE);
    }
	
#ifdef RANDOM_TEST
	CPU_fill_rand(h_emb_table, vocab_size, embedding_size);
#endif    
	
	err = cudaMalloc((void **)&emb_table, size_emb_table);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device emb_table (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	
	
	read_data(h_emb_table, vocab_size, embedding_size, fname[0]);
	////////////////////////////////
		//for(int i=0; i<10; i++)
		//std::cout<<(h_emb_table[i])<<"  ";
	
    err = cudaMemcpy(emb_table, h_emb_table, size_emb_table, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_emb_table from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	
	
	
	

	

	

	int size_t_layer = batch_size*max_seq_len*hidden_size*sizeof(float);
	
	
	err = cudaMalloc((void **)&emb_layer_output, size_t_layer);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device emb_layer_output (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}		
	
	
	err = cudaMalloc((void **)&t_query_layer, size_t_layer);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device t_query_layer (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
	err = cudaMalloc((void **)&t_key_layer, size_t_layer);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device t_key_layer (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&t_value_layer, size_t_layer);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device t_value_layer (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
	
	err = cudaMalloc((void **)&t_context_layer, size_t_layer);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device t_context_layer (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
	
	err = cudaMalloc((void **)&t_attention_scores, batch_size*num_attention_heads*max_seq_len*max_seq_len*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device t_attention_scores (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	
		//attention_scores
	err = cudaMalloc((void **)&attention_scores_up, batch_size*max_seq_len*hidden_size*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device attention_scores_up (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
		exit(EXIT_FAILURE);
	}	
	err = cudaMalloc((void **)&attention_scores_down, batch_size*max_seq_len*hidden_size*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device attention_scores_down (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
		exit(EXIT_FAILURE);
	}	


	//each query-key-value
	float * t_kernel_ptr;
	float * t_kernel_intermediate_ptr;
	float * t_bias_intermediate_ptr;	
	
	float * t_kernel_ptr_dev;
	float * t_bias_ptr_dev;
	
	float * t_layer_dev;
	
	float * t_norm;
	float * t_norm_dev;	
	
	int size_bias = hidden_size * sizeof(float);
	int size_kernel = hidden_size * size_bias;
		

	t_kernel_ptr = (float *)malloc(size_kernel);
	t_norm = (float *)malloc(size_bias);	
	
	int size_bias_intermediate = intermediate_size*sizeof(float);
	int size_kernel_intermediate = hidden_size*size_bias_intermediate;
	int size_intermediate_layer = batch_size*max_seq_len*intermediate_size*sizeof(float);
	
	int size_bias_output_dense = hidden_size*sizeof(float);
	int size_kernel_output_dense = intermediate_size * size_bias_output_dense;
		
	t_kernel_intermediate_ptr = (float *)malloc(size_kernel_intermediate);
	t_bias_intermediate_ptr = (float *)malloc(size_bias_intermediate);

	if (t_kernel_ptr == NULL || t_norm == NULL || t_kernel_intermediate_ptr==NULL ||t_bias_intermediate_ptr==NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors kernel or bias!\n");
		exit(EXIT_FAILURE);
	}
	
	
	
	for(i=0; i<num_hidden_layers; i++)
	{		


		//norm
		err = cudaMalloc((void **)&t_norm_dev, size_bias);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_norm_dev--attention (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
		
		read_data(t_norm, 1, hidden_size, fname[1 + i*12 ]);
		err = cudaMemcpy(t_norm_dev, t_norm, size_bias, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_norm_dev--output_dense from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}
		
		attention_LayerNorm_gamma.push_back(t_norm_dev);
		
		
		err = cudaMalloc((void **)&t_norm_dev, size_bias);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_norm_dev--attention (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
		
		read_data(t_norm, 1, hidden_size, fname[2 + i*12]);

		
		err = cudaMemcpy(t_norm_dev, t_norm, size_bias, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_norm_dev--output_dense from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}
		
		attention_LayerNorm_beta.push_back(t_norm_dev);	
		
		
		
		//query
		err = cudaMalloc((void **)&t_kernel_ptr_dev, size_kernel);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_kernel_ptr_dev--query (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
			
		read_data(t_kernel_ptr, hidden_size, hidden_size, fname[3 + i*12 ]);

		err = cudaMemcpy(t_kernel_ptr_dev, t_kernel_ptr, size_kernel, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_kernel_ptr--query from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
			
		query_kernel.push_back(t_kernel_ptr_dev);

		
		
		//key
		err = cudaMalloc((void **)&t_kernel_ptr_dev, size_kernel);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_kernel_ptr_dev--key (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		

		read_data(t_kernel_ptr, hidden_size, hidden_size, fname[4 + i*12]);
		err = cudaMemcpy(t_kernel_ptr_dev, t_kernel_ptr, size_kernel, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_kernel_ptr--key from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
		
		key_kernel.push_back(t_kernel_ptr_dev);
		
		
		//value
		err = cudaMalloc((void **)&t_kernel_ptr_dev, size_kernel);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_kernel_ptr_dev--value (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		

		read_data(t_kernel_ptr, hidden_size, hidden_size, fname[5 + i*12]);
		err = cudaMemcpy(t_kernel_ptr_dev, t_kernel_ptr, size_kernel, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_kernel_ptr--value from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	

		value_kernel.push_back(t_kernel_ptr_dev);




		//attention_output_transform_kernel
		err = cudaMalloc((void **)&t_kernel_ptr_dev, size_kernel);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_kernel_ptr_dev--value (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		

		read_data(t_kernel_ptr, hidden_size, hidden_size, fname[6 + i*12]);
		err = cudaMemcpy(t_kernel_ptr_dev, t_kernel_ptr, size_kernel, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_kernel_ptr--value from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	

		attention_output_transform_kernel.push_back(t_kernel_ptr_dev);

		
		
		
		//3-layer		
		err = cudaMalloc((void **)&t_layer_dev, size_t_layer);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_layer_dev--query (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
		query_layer.push_back(t_layer_dev);

		
		err = cudaMalloc((void **)&t_layer_dev, size_t_layer);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_layer_dev--key (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
		key_layer.push_back(t_layer_dev);
		
		
		err = cudaMalloc((void **)&t_layer_dev, size_t_layer);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_layer_dev--value (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
		value_layer.push_back(t_layer_dev);



		
		
		//context_layer
		err = cudaMalloc((void **)&t_layer_dev, size_t_layer);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_layer_dev--key (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
		context_layer.push_back(t_layer_dev);
		


		//attention_scores
		err = cudaMalloc((void **)&t_layer_dev, batch_size*max_seq_len*hidden_size*sizeof(float));
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_layer_dev--value (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
		attention_scores.push_back(t_layer_dev);


		
		
		
		//norm
		err = cudaMalloc((void **)&t_norm_dev, size_bias);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_norm_dev--attention (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
		
		read_data(t_norm, 1, hidden_size, fname[7 + i*12]);
		err = cudaMemcpy(t_norm_dev, t_norm, size_bias, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_norm_dev--output_dense from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}
		
		output_LayerNorm_gamma.push_back(t_norm_dev);
		
		
		err = cudaMalloc((void **)&t_norm_dev, size_bias);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_norm_dev--attention (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
		
		read_data(t_norm, 1, hidden_size, fname[8 + i*12]);

		
		err = cudaMemcpy(t_norm_dev, t_norm, size_bias, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_norm_dev--output_dense from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}
		
		output_LayerNorm_beta.push_back(t_norm_dev);
		
		
		
		
		
		//attention_output_intermediate
		err = cudaMalloc((void **)&t_kernel_ptr_dev, size_kernel_intermediate);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_kernel_ptr_dev--attention_output_intermediate (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
		err = cudaMalloc((void **)&t_bias_ptr_dev, size_bias_intermediate);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_bias_ptr_dev--attention_output_intermediate (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
		
		read_data(t_kernel_intermediate_ptr, hidden_size, intermediate_size, fname[9 + i*12]);
		err = cudaMemcpy(t_kernel_ptr_dev, t_kernel_intermediate_ptr, size_kernel_intermediate, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_kernel_ptr--attention_output from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
		
		read_data(t_bias_intermediate_ptr, 1, intermediate_size, fname[10 + i*12]);
		err = cudaMemcpy(t_bias_ptr_dev, t_bias_intermediate_ptr, size_bias_intermediate, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_bias_ptr--attention_output from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}
		attention_output_kernel.push_back(t_kernel_ptr_dev);
		attention_output_bias.push_back(t_bias_ptr_dev);	

		
		
		err = cudaMalloc((void **)&t_layer_dev, size_intermediate_layer);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_layer_dev--attention_output (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}				
		attention_output.push_back(t_layer_dev);	






		//output_dense
		err = cudaMalloc((void **)&t_kernel_ptr_dev, size_kernel_output_dense);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_kernel_ptr_dev--output_dense (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
		err = cudaMalloc((void **)&t_bias_ptr_dev, size_bias_output_dense);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_bias_ptr_dev--output_dense (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}		
		
		read_data(t_kernel_intermediate_ptr, intermediate_size, hidden_size, fname[11 + i*12]);
		err = cudaMemcpy(t_kernel_ptr_dev, t_kernel_intermediate_ptr, size_kernel_output_dense, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_kernel_ptr--output_dense from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}	
		
		read_data(t_norm, 1, hidden_size, fname[12 + i*12]);
		err = cudaMemcpy(t_bias_ptr_dev, t_norm, size_bias_output_dense, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy t_bias_ptr--output_dense from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}
		output_dense_kernel.push_back(t_kernel_ptr_dev);
		output_dense_bias.push_back(t_bias_ptr_dev);	

		

	
		
		
		//int size_output_dense_layer = batch_size*max_seq_len*hidden_size*sizeof(float);
		err = cudaMalloc((void **)&t_layer_dev, size_t_layer);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device t_layer_dev--output_dense (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
			exit(EXIT_FAILURE);
		}				
		layer_output.push_back(t_layer_dev);			
		
	}


	
	
		//norm
	err = cudaMalloc((void **)&pooler_LayerNorm_gamma, size_bias);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device pooler_LayerNorm_gamma (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
		exit(EXIT_FAILURE);
	}		
		
	read_data(t_norm, 1, hidden_size, fname[1 + 12*num_hidden_layers]);
	err = cudaMemcpy(pooler_LayerNorm_gamma, t_norm, size_bias, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy pooler_LayerNorm_gamma from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
		exit(EXIT_FAILURE);
	}
		
	//pooler_LayerNorm_gamma.push_back(t_norm_dev);
		
		
	err = cudaMalloc((void **)&pooler_LayerNorm_beta, size_bias);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device pooler_LayerNorm_beta (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
		exit(EXIT_FAILURE);
	}		
		
	read_data(t_norm, 1, hidden_size, fname[2 + 12*num_hidden_layers]);

		
	err = cudaMemcpy(pooler_LayerNorm_beta, t_norm, size_bias, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy pooler_LayerNorm_beta from host to device (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
		exit(EXIT_FAILURE);
	}
		
	//pooler_LayerNorm_beta.push_back(t_norm_dev);	
	
	
	err = cudaMalloc((void **)&pooler_input, batch_size*hidden_size*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device pooler_input--pooler (error code %s)! for layer %d\n", cudaGetErrorString(err), i);
		exit(EXIT_FAILURE);
	}	

	
	
	err = cudaMalloc((void **)&final_res, vocab_size*batch_size*(sizeof(float)));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device final_res (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }		
	printf("Allocating memory on host and device finished\n");
	
	
	//h_final_res = (float *)malloc(vocab_size * batch_size *  sizeof(float));
	
	
	
    err = cudaMalloc((void **)&d_C_res_pos, batch_size * topk_num *  sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_C_res_pos (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMalloc((void **)&d_C_res_dis, batch_size * topk_num *  sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_C_res_dis (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	
	
	

	h_res_vle = (float *)malloc(batch_size * topk_num *  sizeof(float));;
	h_res_idx = (int *)malloc(batch_size * topk_num *  sizeof(int));







	free(t_kernel_ptr); 
	free(t_norm); 
	
		
	free(t_kernel_intermediate_ptr);
	free(t_bias_intermediate_ptr);	
}



	//batch_ids==sen ids   mask_len==sen length 
void transformer::forward(const int *batch_ids, const int *mask_len, bool use_softmax)
{
	//mask_length = mask_len;
	size_t size_input_ids = batch_size * max_seq_len * sizeof(int);
	cudaMemcpy(mask_length_dev, mask_len, batch_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(input_ids_dev, batch_ids, size_input_ids, cudaMemcpyHostToDevice);
	//cudaMemcpy(input_token_type_ids_dev, input_token_type_ids, size_input_ids, cudaMemcpyHostToDevice);
	
	
#ifdef SHOW_TIMECOST	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif

	//embedding_lookup(input_ids_dev, batch_size, max_seq_len, vocab_size, embedding_size);

	
	
	
	embedding_lookup_posemb_cal(input_ids_dev, batch_size, max_seq_len, embedding_size, 1.0f,  10000.0f);
	

	
	/*if(!use_tokentype)
		Embedding_PosEmb_Add(input_ids_dev, batch_size, max_seq_len, vocab_size, embedding_size);
	else
		Embedding_PosEmb_TokenType_Add(input_ids_dev, input_token_type_ids_dev, batch_size, max_seq_len, vocab_size, embedding_size);*/

//////////////////////////////////////////////////////////////




	
	//layer_norm(emb_layer_output, emb_layer_output, emb_LayerNorm_gamma, emb_LayerNorm_beta, embedding_size, batch_size*max_seq_len);




	
	
	
	

	for(int i=0; i<num_hidden_layers; i++)
	{
		if(i==0)
			transformer_input = emb_layer_output;
		else
			transformer_input = layer_output[i-1];
		
		attention_layer(transformer_input, transformer_input, num_attention_heads, hidden_size/num_attention_heads, i,
						batch_size*max_seq_len, hidden_size, batch_size*max_seq_len, hidden_size);
						
						

	}

	
	//layer_norm(layer_output[num_hidden_layers-1], layer_output[num_hidden_layers-1], pooler_LayerNorm_gamma, pooler_LayerNorm_beta, embedding_size, batch_size*max_seq_len);
	

	
	
	get_any_token(layer_output[num_hidden_layers-1], pooler_input, batch_size, max_seq_len, hidden_size, mask_length_dev);

	
	layer_norm(pooler_input, pooler_input, pooler_LayerNorm_gamma, pooler_LayerNorm_beta, embedding_size, batch_size);
	
/*{
	float * ttres1 = (float *)malloc(embedding_size*max_seq_len*batch_size*(sizeof(float)));
	cudaMemcpy(ttres1, pooler_input, embedding_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
	
	cout<<endl;
	for(int a1=0; a1<4; a1++)
	{
		for(int a3=0; a3<10; a3++)
		{
			cout<<ttres1[a1*embedding_size + a3]<<" ";
		}
		cout<<endl;
	}	
}*/	
	
	
	//dense_gpu(handle, pooler_input, emb_table, emb_table, final_res, hidden_size, vocab_size, batch_size, "no_bias");
	
	
	const float alf = (1.0f);
    const float bet = (0.0f);
    const float *alpha = &alf;
    const float *beta = &bet;
	cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, vocab_size, batch_size, hidden_size, alpha, emb_table, hidden_size, pooler_input, hidden_size, beta, final_res, vocab_size);

	if(use_softmax)
		softmax_big(final_res, final_res, batch_size, vocab_size);
	
	topk_fun(final_res, batch_size, vocab_size, topk_num, d_C_res_dis, d_C_res_pos);
	cudaMemcpy(h_res_vle, d_C_res_dis, batch_size * topk_num *  sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_res_idx, d_C_res_pos, batch_size * topk_num *  sizeof(int), cudaMemcpyDeviceToHost);	

	//topk_bucket(final_res, d_C_res_dis, d_C_res_pos, d_len_res, batch_size, vocab_size, MAX_TOPK_NUM);



	
	//cudaMemcpy(h_C_res_dis, d_C_res_dis, batch_size * MAX_TOPK_NUM *  sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_C_res_pos, d_C_res_pos, batch_size * MAX_TOPK_NUM *  sizeof(int), cudaMemcpyDeviceToHost);	
	//cudaMemcpy(h_len_res, d_len_res, batch_size * sizeof(int), cudaMemcpyDeviceToHost);	


	/*for(int a1=0; a1<1; a1++)
	{
		std::cout<<"batch--"<<a1<<std::endl;
		for(int a2=0; a2<MAX_TOPK_NUM; a2++)
			std::cout<<h_C_res_pos[a1*3+a2]<<"--"<<h_C_res_dis[a1*3+a2]<<"   ";
		std::cout<<std::endl;
	}*/	
	
	//get_topk_heap(h_C_res_pos, h_C_res_dis, h_len_res, batch_size, topk_num, MAX_TOPK_NUM, h_res_idx, h_res_vle);
	


#ifdef SHOW_TIMECOST
	cudaEventRecord(stop,0);
	cudaEventSynchronize( stop );
	float costtime;
	cudaEventElapsedTime(&costtime,start,stop);
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Test PASSED costtime=%f\n", costtime);
#endif		
	
/*	for(int a1=0; a1<1; a1++)
	{
		std::cout<<"batch--"<<a1<<std::endl;
		for(int a2=0; a2<topk_num; a2++)
			std::cout<<h_res_idx[a1*topk_num+a2]<<"--"<<h_res_vle[a1*topk_num+a2]<<"  ";
		std::cout<<std::endl;
	}*/

	
/*	float * ttres = (float *)malloc(vocab_size*batch_size*(sizeof(float)));
	cudaMemcpy(ttres, final_res, vocab_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
	
	int * idx_arr = (int *)malloc(3*batch_size*(sizeof(int)));
	for(int a1=0; a1<batch_size; a1++)
	{
		for(int a2=0; a2<3; a2++)
		{
			float max=(-1000000.0f);
			for(int a3=0; a3<vocab_size; a3++)
			{
				int pos = a1*vocab_size + a3;
				if((ttres[pos])>max)
				{
					max=(ttres[pos]);
					idx_arr[a1*3+a2] = a3;
				}
			}
			//cout<<max<<" ";
			ttres[a1*vocab_size + idx_arr[a1*3+a2]] = (-10000000.0f);
		}
		//cout<<endl;
	}
	for(int a1=0; a1<batch_size; a1++)
	{
		std::cout<<"batch--"<<a1<<std::endl;
		for(int a2=0; a2<3; a2++)
			std::cout<<idx_arr[a1*3+a2]<<"  ";
		std::cout<<std::endl;
	}*/
	

}

struct cmp
{
	template<typename T, typename U>
	bool operator()(T const& left, U const &right) 
	{
		if (left.second > right.second) return true;
		return false;
	}
};


void transformer::get_topk_heap(int *idx, float *idx_vle, int *len, int batchsize, int topk, int max_ele, int *res_idx, float *res_vle)
{
	for(int ii=0; ii<batchsize; ii++)
	{
		std:: priority_queue<pair<int, float>, vector<pair<int, float> >, cmp > tmp_res_heap;
		for(int i=0; i<topk && i < len[ii]; i++)
		{
			std::pair<int, float> a(idx[ii*max_ele + i], idx_vle[ii*max_ele + i]);
			tmp_res_heap.push(a);
		}
		
		for (int i =topk; i<len[ii]; i++)
		{
			if(idx_vle[ii*max_ele + i] > tmp_res_heap.top().second)
			{
				tmp_res_heap.pop();
				std::pair<int, float> a(idx[ii*max_ele + i], idx_vle[ii*max_ele + i]);
				tmp_res_heap.push(a);
			}			
		}
		
		int cc=0;
		while (cc<topk && !tmp_res_heap.empty())
		{
            res_idx[ii*topk + cc] = tmp_res_heap.top().first;
			res_vle[ii*topk + cc] = tmp_res_heap.top().second;
            tmp_res_heap.pop();
			cc++;
		}
		
	}

}



__global__ void softmax_big_kernel(float *i_mat, float* o_mat, int i_nRows, int i_nCols, int dpt)
{
	__shared__ float sum_arr[32];
	int bx= blockIdx.x;
	int tx= threadIdx.x;
	int base_idx= bx * i_nCols;
	int col_idx= tx;
	float sum_aa = 0.0f;
	int i, pos;
	for(i = 0; i < dpt; i++)
	{	
		if(col_idx < i_nCols)
		{	
			sum_aa += expf(i_mat[base_idx+col_idx]*0.1f);	
		}
		col_idx += REDUCE_SIZE;
	}
	__syncthreads();
	for (int i = 16; i >= 1; i /= 2)
	{
	#ifndef SMFOR35
		sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
	#else
		sum_aa += __shfl_xor(sum_aa, i, 32);
	#endif
	}	
	if(tx%32==0)
	{
		sum_arr[tx/32] = sum_aa;
	}	
	__syncthreads();
	if(tx<32)
	{
		sum_aa = sum_arr[tx];
		for (int i = 16; i >= 1; i /= 2)
		{
		#ifndef SMFOR35
			sum_aa += __shfl_xor_sync(0xffffffff, sum_aa, i, 32);
		#else
			sum_aa += __shfl_xor(sum_aa, i, 32);
		#endif
		}	
		__syncthreads();
		if(tx==0)
		{
			sum_arr[0] = sum_aa;
		}	
	}
	__syncthreads();
	col_idx = tx;
	for(i = 0; i < dpt; i++)
	{
		//col_idx += blockDim.x;
		if(col_idx < i_nCols)
		{	
			o_mat[base_idx+col_idx] = expf(i_mat[base_idx+col_idx]*0.1f)/sum_arr[0];	
		}
		col_idx += REDUCE_SIZE;
	}	
}

void transformer::softmax_big(float *i_mat, float* o_mat, int i_nRows, int i_nCols)
{
	int num_thread=REDUCE_SIZE;
	int dpt = i_nCols / num_thread;
	if(i_nCols % num_thread != 0) 
		dpt++;	
	softmax_big_kernel<<<i_nRows, num_thread>>>(i_mat, o_mat, i_nRows, i_nCols, dpt);
	//getLastCudaError("Error in Calling 'kernel'--output_max");
}






transformer::~transformer()
{
	cudaFree(emb_table);
	cudaFree(emb_layer_output);	
	cudaFree(t_query_layer);
	
	
	cudaFree(t_key_layer);
	cudaFree(t_value_layer);
	cudaFree(input_ids_dev);
	cudaFree(t_context_layer);
	cudaFree(attention_scores_down);
	cudaFree(attention_scores_up);
	cudaFree(t_attention_scores);
	cudaFree(pooler_LayerNorm_beta);
	cudaFree(pooler_LayerNorm_gamma);
	cudaFree(pooler_input);

	cudaFree(mask_length_dev);
	cudaFree(final_res);
	
	cudaFree(d_C_res_dis);
	cudaFree(d_C_res_pos);
	//cudaFree(d_len_res);
	
	for(int i=0; i<num_hidden_layers; i++)
	{
		cudaFree(query_kernel[i]);
		cudaFree(key_kernel[i]);
		cudaFree(value_kernel[i]);
		cudaFree(query_layer[i]);
		cudaFree(key_layer[i]);
		cudaFree(value_layer[i]);
		cudaFree(attention_output_transform_kernel[i]);
		cudaFree(context_layer[i]);
		cudaFree(attention_scores[i]);
		cudaFree(attention_output_kernel[i]);
		cudaFree(attention_output_bias[i]);
		cudaFree(attention_output[i]);
		cudaFree(output_dense_kernel[i]);
		cudaFree(output_dense_bias[i]);
		cudaFree(layer_output[i]);
		cudaFree(attention_LayerNorm_beta[i]);
		cudaFree(attention_LayerNorm_gamma[i]);
		cudaFree(output_LayerNorm_beta[i]);
		cudaFree(output_LayerNorm_gamma[i]);
	}
	
	
	
	
	free(h_emb_table);
	//free(h_final_res);
	
	free(h_res_idx);
	free(h_res_vle);
	
	//free(h_C_res_dis);
	//free(h_C_res_pos);
	//free(h_len_res);
	
}



/*extern "C" 
{  
    transformer myobj;  

	//n_vs==vocabsize  n_es==embbeding_size  n_hs=hidden_size  n_hLayer==num_attention_layers  n_aHead==num_heads
	//n_is==num_filter_size  n_bs==batch_size 
	//maxseqlen==max_seq_len  ttk==topk_num
    void init(int n_vs, int n_es, int n_hs, int n_hLayer, int n_aHead, int n_is, int mpe, int n_tvs,  int n_bs, int maxseqlen, int n_labels, int ttk) 
	{
		vector<string> fname_base;
		std::ifstream ifile;
		ifile.open("weights.name");
		std::string line;
		std::string dir="./data/";
		while (std::getline(ifile, line)) 
		{
			//std::cout << line << std::endl;
			line = dir+line;
			fname_base.push_back(line);
		}
		ifile.close();
		myobj.Initialize(7, n_vs,  n_es,  n_hs,  n_hLayer,  n_aHead,  n_is, n_bs,  maxseqlen, ttk, fname_base);
    }  
	
	
	//sen==sen ids   len==sen length 
	//h_res_dis==probability of results  h_res_pos==ids of results
    void forward_cal(int *sen,  int* len, float *h_res_dis, int *h_res_pos) 
	{ 
		myobj.forward(sen, len, false);
	
	h_res_dis = myobj.get_res_vle();
	h_res_pos = myobj.get_res_idx();
	
    }  
}*/



int main(void)
{
	vector<string> fname_base;
	std::ifstream ifile;
	ifile.open("weights.name");
    std::string line;
	std::string dir="./data/";
    while (std::getline(ifile, line)) {
        //std::cout << line << std::endl;
		line = dir+line;
		fname_base.push_back(line);
    }
    ifile.close();

	int n_vs = 33276;//271480;//21128;
	int n_es = 512;
	int n_hs = 512;
	int n_hLayer = 4;
	int n_aHead = 8;
	int n_is = 512*4;

	int n_bs = 128;
	int maxseqlen = 64;
	//int n_labels = 195429;//211964;//94696;
	int ttk=10;


	//int ri[64]={1971, 22234, 9022, 12445, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	//int ri[64]={21625,1524,33273,7145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	//int ri[64]={1971, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	//int ri[64]={1971, 22234, 9022, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	int ri[64]={8, 21625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	//{271475, 75, 271476, 4572, 271476, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ;
	//{101, 1506, 1506, 1506, 671, 833, 5314, 872, 7741, 6395, 4772, 102, 679, 3221, 4510, 6228, 1196, 4638, 6929, 702, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	//int rto[64]={0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	//{0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	//{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	
	int *len = new int[n_bs];
	for(int i=0; i<n_bs; i++)
		len[i] = 2;
	int *sen = new int[n_bs*maxseqlen];
	//int *tids = new int[n_bs*maxseqlen];
	for(int i=0;i<n_bs; i++)
	{
		for(int j=0; j<maxseqlen; j++)
		{
			sen[i*maxseqlen +j] = ri[j];
			//tids[i*maxseqlen +j] = rto[j];
		}
	}
	
	transformer myobj;
	myobj.Initialize(7, n_vs,  n_es,  n_hs,  n_hLayer,  n_aHead,  n_is, n_bs,  maxseqlen, ttk, fname_base);
	//myobj.forward(sen, len, true);
	myobj.forward(sen, len, false);
	
	float *h_res_dis = myobj.get_res_vle();
	int *h_res_pos = myobj.get_res_idx();
	for(int xc=0; xc<3; xc++)
	{
		cout<<"res-"<<xc<<" =="<<endl;
		for(int cv=0; cv<ttk; cv++)
		{
			cout<<h_res_dis[xc*ttk +cv]<<" "<<h_res_pos[xc*ttk +cv]<<endl;
		}	
	}
	return 0;
}

