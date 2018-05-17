/*
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	MapReduce Framework on GPU and CPU cluster
	Code Name: Panda
	File: Panda.h 
	First Version:		2012-07-01 V0.1
	Last  Version:		2012-09-01 V0.3
	Last Updates:		2017-12-24 V0.60

	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
*/

/*
#ifdef WIN32 
#include <windows.h> 
#endif 
#include <pthread.h>
*/
#include <cuda.h>
#ifndef __PANDA_H__
#define __PANDA_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <assert.h>
#include <time.h>
#include <stdarg.h>
#include <pthread.h>
#include <iostream>
#include <vector>

#define _DEBUG		0x01
#define _WARN		0x02
#define _ERROR		0x03
#define _DISKLOG	0x04

#define CEIL(n,m) (n/m + (int)(n%m !=0))
#define THREAD_CONF(grid, block, gridBound, blockBound) do {\
	    block.x = blockBound;\
	    grid.x = gridBound; \
		if (grid.x > 65535) {\
		   grid.x = (int)sqrt((double)grid.x);\
		   grid.y = CEIL(gridBound, grid.x); \
		}\
	}while (0)

#define THREADS_PER_BLOCK	(blockDim.x * blockDim.y)
#define BLOCK_ID		(gridDim.y	* blockIdx.x  + blockIdx.y)
#define THREAD_ID		(blockDim.x	* threadIdx.y + threadIdx.x)
#define TID			(BLOCK_ID * THREADS_PER_BLOCK + THREAD_ID)
//#define TID (BLOCK_ID * blockDim.x + THREAD_ID)

//NOTE NUM_THREADS*NUM_BLOCKS > STRIDE !
#define NUM_THREADS			256
#define NUM_BLOCKS			4
#define STRIDE				32
#define THREAD_BLOCK_SIZE	16

#define SHARED_BUFF_LEN			204800
#define CPU_SHARED_BUFF_SIZE	40960
#define GPU_SHARED_BUFF_SIZE	40960

#define _MAP		-1
#define _COMBINE	-2
#define _SHUFFLE	-3
#define _REDUCE		-4

extern int gCommRank;

#ifdef _DEBUG
#define ShowLog(...) do{printf("[%d]",gCommRank);printf("[%s]\t",__FUNCTION__);printf(__VA_ARGS__);printf("\n");fflush(stdout);}while(0)
#define ShowLog2(...) do{printf("[%s]\t",__FUNCTION__);printf(__VA_ARGS__);printf("\n");}while(0)
#else
#define ShowLog(...) //do{printf(__VA_ARGS__);printf("\n");}while(0)
#endif

#ifdef _DISKLOG
#define DoLog2Disk(...) do{ FILE *fptr;	fptr = fopen("panda.log","a"); fprintf(fptr,"[PandaDiskLog]\t\t"); fprintf(fptr,__VA_ARGS__);fprintf(fptr,"\n");fclose(fptr);}while(0)
#else
#define DoLog2Disk(...) 
#endif

extern "C"
void DoDiskLog(const char *str);

#ifdef _ERROR
#define ShowError(...) do{printf("[%d]",gCommRank);printf("[%s]\t",__FUNCTION__);printf(__VA_ARGS__);printf("\n");fflush(stdout);}while(0)
#else
#define ShowError(...)
#endif

#ifdef _ERROR
#define GpuShowError(...) do{printf("[%s]\t",__FUNCTION__);printf(__VA_ARGS__);printf("\n");}while(0)
#else
#define GpuShowError(...)
#endif

#ifdef _WARN
#define ShowWarn(...) do{printf("[%s]\t",__FUNCTION__);printf(__VA_ARGS__);printf("\n");}while(0)
#else
#define ShowWarn(...) 
#endif

struct panda_cpu_context;
struct panda_node_context;
struct panda_gpu_card_context;
struct panda_gpu_context;

struct keyval_t
{
   void *key;
   void *val;
   int keySize;
   int valSize;
   int task_idx;//map_task_idx, reduce_task_idx
};// keyval_t;

struct job_configuration
{		
	bool auto_tuning;
	bool local_combiner;
	bool iterative_support;

	int num_input_record;
	keyval_t * input_keyval_arr;
	
	int num_mappers;
	int num_reducers;
	
	int num_gpu_core_groups;
	int num_gpu_card_groups;
	int num_cpus_groups;

	int num_cpus_cores;
	int auto_tuning_sample_rate;
		
};//job_configuration;

extern "C"
void *RunPandaCPUCombinerThread(void *ptr);
extern "C"
void ExecutePandaGPUCombiner(panda_gpu_context *pgc);
extern "C"
void ExecutePandaGPUSort(panda_gpu_context *pgc);
extern "C"
void ExecutePandaShuffleMergeGPU(panda_node_context *d_g_state_1, panda_gpu_context *d_g_state_0);
extern "C"
void ExecutePandaCPUSort(panda_cpu_context *pcc, panda_node_context *pnc);
extern "C"
void ExecutePandaReduceTasksOnGPU(panda_gpu_context *pgc);
extern "C"
void* ExecutePandaCPUMapThread(void* thread_info);
extern "C"
void StartPandaGPUMapPartitioner(panda_gpu_context pgc, dim3 grids, dim3 blocks);
extern "C"
void RunGPUMapTasksHost(panda_gpu_context pgc, int curIter, int totalIter, dim3 grids, dim3 blocks);
extern "C"
void ExecutePandaSortBucket(panda_node_context *pnc);
extern "C"
void ExecutePandaCPUCombiner(panda_cpu_context *pcc);
//void __checkCudaErrors(cudaError err, const char *file, const int line );

int cudaMemGetInfo(void *, size_t *);
int cudaMemset(void *,int,int);
void PandaLaunchMapPartitionerOnGPU(panda_gpu_context,dim3,dim3);
void PandaLaunchMapTasksOnGPU(panda_gpu_context, int, int , dim3,dim3);
double PandaTimer();

struct keyval_pos_t
{

   int keySize;
   int valSize;
   int keyPos;
   int valPos;

   int task_idx;
   int next_idx;
   
};// keyval_pos_t;

//typedef 
struct val_pos_t
{
   int valSize;
   int valPos;
};// val_pos_t;

//typedef
struct sorted_keyval_pos_t
{
   int keySize;
   int keyPos;

   int val_arr_len;
   val_pos_t * val_pos_arr;
};// sorted_keyval_pos_t;

//two direction - bounded share buffer
// from left to right  key val buffer
// from right to left  keyval_t buffer
struct keyval_arr_t
{
   
   int *shared_arr_len;
   int *shared_buddy;
   int shared_buddy_len;

   char *shared_buff;
   int *shared_buff_len;
   int *shared_buff_pos;

   //int keyval_pos;
   int arr_len;
   keyval_pos_t *arr;
   keyval_t *cpu_arr;

};// keyval_arr_t;

//used for sorted or partial sorted values
//typedef 
struct val_t
{
   void * val;
   int valSize;
};// val_t;

//typedef 
struct keyvals_t
{
   void * key;
   int keySize;
   int val_arr_len;
   val_t * vals;
};// keyvals_t;

struct panda_cpu_task_info_t {	
	
	int tid;				//accelerator group
	int num_cpus_cores;			//num of processors
	char device_type;			//
	panda_cpu_context *pcc;			//gpu_context  cpu_context
	panda_node_context *pnc;		//
	void *cpu_job_conf; //depricated	
	int start_row_idx;
	int end_row_idx;

};// panda_cpu_task_info_t;

struct panda_gpu_card_context
{
};// gpu_card_context;


//typedef
struct panda_gpu_context
{	

	struct{

	void *d_input_keys_shared_buff;
	void *d_input_vals_shared_buff;
	keyval_pos_t *d_input_keyval_pos_arr;
	//data for input results
	int num_input_record;
	keyval_t * h_input_keyval_arr;
	keyval_t * d_input_keyval_arr;

	} input_key_vals;

	struct{

	//data for intermediate results
	int *d_intermediate_keyval_total_count;
	int d_intermediate_keyval_arr_arr_len;			//number of elements of d_intermediate_keyval_arr_arr
	//keyval_arr_t *d_intermediate_keyval_arr_arr;	//data structure to store intermediate keyval pairs in device
	keyval_arr_t **d_intermediate_keyval_arr_arr_p;	
	keyval_t* d_intermediate_keyval_arr;				//data structure to store flattened intermediate keyval pairs
	void *d_intermediate_keys_shared_buff;
	void *d_intermediate_vals_shared_buff;
	keyval_pos_t *d_intermediate_keyval_pos_arr;
	void *h_intermediate_keys_shared_buff;
	void *h_intermediate_vals_shared_buff;
	keyval_pos_t *h_intermediate_keyval_pos_arr;

	} intermediate_key_vals;
	
	struct{
	
	//data for sorted intermediate results
	int d_sorted_keyvals_arr_len;
	void *h_sorted_keys_shared_buff;
	void *h_sorted_vals_shared_buff;
	int totalKeySize;
	int totalValSize;
	sorted_keyval_pos_t *h_sorted_keyval_pos_arr;
	
	void *d_sorted_keys_shared_buff;
	void *d_sorted_vals_shared_buff;
	keyval_pos_t *d_keyval_pos_arr;
	int *d_pos_arr_4_sorted_keyval_pos_arr;
	
	}sorted_key_vals;
	
	struct{
	
	int d_reduced_keyval_arr_len;
	keyval_t* d_reduced_keyval_arr;
	
	} reduced_key_vals;
	
	struct{
	
	int h_reduced_keyval_arr_len;
	keyval_t* h_reduced_keyval_arr;
	void *h_KeyBuff;
	void *h_ValBuff;
	void *d_KeyBuff;
	void *d_ValBuff;
	int totalKeySize;
	int totalValSize;
	
	} output_key_vals;
	
};// panda_gpu_context;


//typedef
struct panda_cpu_context
{	
	
	int num_cpus_cores;
	pthread_t  *panda_cpu_task_thread;
	panda_cpu_task_info_t *panda_cpu_task_thread_info;

	struct{
	
	void *input_keys_shared_buff;
	void *input_vals_shared_buff;
	keyval_pos_t *input_keyval_pos_arr;
	int num_input_record;
	keyval_t * input_keyval_arr;
	
	}input_key_vals;
	
	struct{
	
	int *intermediate_keyval_total_count;
	int intermediate_keyval_arr_arr_len;			//number of elements of d_intermediate_keyval_arr_arr
	keyval_arr_t *intermediate_keyval_arr_arr_p;	
	keyval_t* intermediate_keyval_arr;				//data structure to store flattened intermediate keyval pairs
	
	void *intermediate_keys_shared_buff;
	void *intermediate_vals_shared_buff;
	keyval_pos_t *intermediate_keyval_pos_arr;
	
	} intermediate_key_vals;

	struct{

	int sorted_keyvals_arr_len;
	keyvals_t		*sorted_intermediate_keyvals_arr;
	void *h_sorted_keys_shared_buff;
	void *h_sorted_vals_shared_buff;
	int totalKeySize;
	int totalValSize;

	}sorted_key_vals;

	struct{

	//data for reduce results
	int reduced_keyval_arr_len;
	keyval_t* reduced_keyval_arr;

	} reduced_key_vals;

	struct{
	
	int reduced_keyval_arr_len;
	keyval_t* reduced_keyval_arr;
	void *KeyBuff;
	void *ValBuff;
	int totalKeySize;
	int totalValSize;
	
	} output_key_vals;
};// panda_cpu_context;


struct panda_node_context
{				
	keyval_t		*input_keyval_arr;
	keyval_arr_t	*intermediate_keyval_arr_arr_p;
			
	struct{	
			
	//data for sorted intermediate results
	int sorted_keyvals_arr_len;
	int sorted_keyval_arr_max_len;
	keyvals_t		*sorted_intermediate_keyvals_arr;
	void *h_sorted_keys_shared_buff;
	void *h_sorted_vals_shared_buff;
	int totalKeySize;
	int totalValSize;
			
	}sorted_key_vals;
			
	struct{	
			
	//data for sending out to remote compute node
	int numBuckets;
	int * keyBuffSize;
	int * valBuffSize;
	std::vector<char * > savedKeysBuff, savedValsBuff;
	std::vector<int  * > counts, keyPos, valPos, keySize, valSize;
	//count[0,1,2,3]  numElements|elementsCapacity|keyPos[elementsCapacity]|valPos[elementsCapacity]
			
	}buckets;

	struct{

	std::vector<char * > savedKeysBuff, savedValsBuff;
	std::vector<int  * > counts, keyPos, valPos, keySize, valSize;

	} recv_buckets;
	
	int num_cpus_groups;
	int num_gpu_core_groups;
	int num_gpu_card_groups;
	int num_all_dev_groups;
	
	struct cpu_context 			*cpu_context;
	struct panda_gpu_context 	*gpu_core_context;
	struct gpu_context 			*gpu_card_context;
	
	float cpu_ratio;
	float gpu_core_ratio;
	float gpu_card_ratio;
	
};

struct panda_runtime_context
{	
	keyval_t		*input_keyval_arr;
	keyval_arr_t	*intermediate_keyval_arr_arr_p;
	keyvals_t		*sorted_intermediate_keyvals_arr;
	int 			sorted_keyvals_arr_len;
	
	int num_cpus_groups;
	int num_gpu_core_groups;
	int num_gpu_card_groups;
	
	int num_all_dev_groups;
};


#define GPU_CORE_ACC			0x01
#define GPU_CARD_ACC			0x05
#define CPU_ACC				0x02
#define CELL_ACC			0x03
#define FPGA_ACC			0x04
	 
void PandaEmitMapOutputOnCPU(void *key,void *val,int keySize,int valSize,panda_cpu_context *pcc,int map_task_idx);
void PandaEmitCombinerOutputOnCPU(void *key, void *val, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx);
void PandaEmitReduceOutputOnCPU(void*	key, void*	val, int	keySize, int	valSize, panda_cpu_context *pcc);


extern "C"
void PandaEmitCPUMapOutput(void *key, void * val, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx);
extern "C"
void PandaCPUEmitCombinerOutput(void *key, void *val, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx); 
extern "C"
void PandaCPUEmitReduceOutput (void* key, void * val, int keySize, int valSize, panda_cpu_context *pcc);

__device__ void PandaGPUEmitMapOutput(void *key, void *val, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx);
__device__ void PandaGPUEmitCombinerOutput(void* key, void * val, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx);
__device__ void PandaGPUEmitReduceOutput(void * key, void *val, int keySize, int valSize, panda_gpu_context *pgc);



__global__ void copyDataFromDevice2Host1(panda_gpu_context pgc);
__global__ void copyDataFromDevice2Host2(panda_gpu_context pgc);
__global__ void copyDataFromDevice2Host4Reduce(panda_gpu_context pgc);
__global__ void PandaReducePartitioner(panda_gpu_context pgc);

void *PandaThreadLaunchCombinerOnCPU(void *ptr);
void *PandaThreadExecuteMapOnCPU(void* thread_info);

void PandaExecuteShuffleMergeOnGPU(panda_node_context *pnc, panda_gpu_context *pgc);

void AddReduceTaskOnGPU(panda_gpu_context* pgc, panda_node_context *pnc, int start_task_id, int end_task_id);
void AddReduceTaskOnGPUCard(panda_gpu_card_context* pgc, panda_node_context *pnc, int start_task_id, int end_task_id);
void AddReduceTaskOnCPU(panda_cpu_context* pgc, panda_node_context *pnc, int start_task_id, int end_task_id);

//void PandaLaunchMapPartitionerOnGPU(panda_gpu_context pgc, dim3 grids, dim3 blocks);
void PandaExecuteReduceTasksOnGPU(panda_gpu_context *pgc);
void PandaExecuteReduceTasksOnGPUCard(panda_gpu_card_context *pgcc);

void PandaExecuteCombinerOnGPU(panda_gpu_context *pgc);
void PandaExecuteCombinerOnGPUCard(panda_gpu_card_context *pgcc);
void PandaExecuteCombinerOnCPU(panda_cpu_context *pcc);

void PandaExecuteSortOnGPUCard(panda_gpu_card_context *pgcc, panda_node_context *pnc);
void PandaExecuteSortOnCPU(panda_cpu_context *pcc, panda_node_context *pnc);
void PandaExecuteSortOnGPU(panda_gpu_context *pgc);

void PandaExecuteSortBucketOnCPU(panda_node_context *pnc);
void ExecutePandaCPUReduceTasks(panda_cpu_context *pcc);
int getCPUCoresNum();
int getGPUCoresNum();

panda_gpu_context		*CreatePandaGPUContext();
panda_cpu_context		*CreatePandaCPUContext();
panda_gpu_card_context		*CreatePandaGPUCardContext();

#endif //__PANDA_H__
