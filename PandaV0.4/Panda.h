/*
 
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: Panda.h 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

/*
#ifdef WIN32 
#include <windows.h> 
#endif 
#include <pthread.h>
*/

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

#define _DEBUG 0x01
#define _WARN 0x02
#define _ERROR 0x03
#define _DISKLOG 0x04

#define CEIL(n,m) (n/m + (int)(n%m !=0))
#define THREAD_CONF(grid, block, gridBound, blockBound) do {\
	    block.x = blockBound;\
	    grid.x = gridBound; \
		if (grid.x > 65535) {\
		   grid.x = (int)sqrt((double)grid.x);\
		   grid.y = CEIL(gridBound, grid.x); \
		}\
	}while (0)

#define THREADS_PER_BLOCK (blockDim.x*blockDim.y)
#define BLOCK_ID	(gridDim.y	* blockIdx.x  + blockIdx.y)
#define THREAD_ID	(blockDim.x	* threadIdx.y + threadIdx.x)
#define TID (BLOCK_ID * THREADS_PER_BLOCK + THREAD_ID)
//#define TID (BLOCK_ID * blockDim.x + THREAD_ID)

//NOTE NUM_THREADS*NUM_BLOCKS > STRIDE !
#define NUM_THREADS	256
#define NUM_BLOCKS	4
#define STRIDE	32
#define THREAD_BLOCK_SIZE 16

#define SHARED_BUFF_LEN 204800
#define CPU_SHARED_BUFF_SIZE 40960000

#define _MAP		-1
#define _COMBINE	-2
#define _SHUFFLE	-3
#define _REDUCE		-4

extern "C"
double PandaTimer();

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

extern "C"
void __checkCudaErrors(cudaError err, const char *file, const int line );


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

//used for unsorted values
//typedef
struct keyval_t
{
   void *key;
   void *val;
   int keySize;
   int valSize;
   int task_idx;//map_task_idx, reduce_task_idx
};// keyval_t;

//two direction - bounded share buffer
// from left to right  key val buffer
// from right to left  keyval_t buffer

//typedef

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
		
/*
job_configuration -> gpu core job
                     gpu card job
					 cpu job
*/

//typedef 
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

	//double cpu_ratio;
	//int matrix_size;
		
};//job_configuration;
			
//typedef 
struct panda_cpu_task_info_t {
							
	int tid;			//accelerator group id
	int num_cpus_cores;		//num of processors
	char device_type;		
	void *d_g_state;		//gpu_context  cpu_context
	void *cpu_job_conf;
	
	int start_row_idx;
	int end_row_idx;

};// panda_cpu_task_info_t;

struct cpu_context
{
	bool iterative_support;		
	bool local_combiner;
	int cpu_group_id;
	int num_input_record;
	int num_cpus_cores;

	keyval_t *input_keyval_arr;
	keyval_arr_t *intermediate_keyval_arr_arr_p;
	keyvals_t *sorted_intermediate_keyvals_arr;
	int sorted_keyvals_arr_len;
	int *intermediate_keyval_total_count;

	pthread_t  *panda_cpu_task;
	panda_cpu_task_info_t *panda_cpu_task_info;

} ;

//typedef
struct gpu_card_context
{		

	bool iterative_support;		
	bool local_combiner;

	int gpu_group_id;		
	int gpu_id;
	int num_input_record;	
	

	int num_gpu_card_groups;			
							
	keyval_t *input_keyval_arr;
	keyval_arr_t *intermediate_keyval_arr_arr_p;
	keyvals_t *sorted_intermediate_keyvals_arr;
	int sorted_keyvals_arr_len;
	int *intermediate_keyval_total_count;
							
	pthread_t  *panda_gpu_task;
	panda_cpu_task_info_t *panda_cpu_task_info;
						
};// gpu_card_context;

struct gpu_context
{	

  bool iterative_support;
  bool local_combiner;

  int gpu_id;   //assigned gpu device id used for resource allocation
  int num_gpu_core_groups;
  //int num_gpu_card_groups;//added

  int num_mappers;
  int num_reducers;

  void *d_input_keys_shared_buff;
  void *d_input_vals_shared_buff;
  keyval_pos_t *d_input_keyval_pos_arr;

  //data for input results
  int num_input_record;

  keyval_t * h_input_keyval_arr;
  keyval_t * d_input_keyval_arr;

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
					
  //data for reduce results
  int d_reduced_keyval_arr_len;
  keyval_t* d_reduced_keyval_arr;

  int *keyval_pairs;
  
};// gpu_context;

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

	}input_key_vals;


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
					
  //data for reduce results
  int d_reduced_keyval_arr_len;
  keyval_t* d_reduced_keyval_arr;
  int *keyval_pairs;

	} reduced_key_vals;
  
};// panda_gpu_context;

//typedef 
struct panda_context{

	keyval_t * input_keyval_arr;
	keyval_arr_t *intermediate_keyval_arr_arr_p;
	keyvals_t * sorted_intermediate_keyvals_arr;
	int sorted_keyvals_arr_len;
	
	int num_cpus_groups;
	int num_gpu_core_groups;
	int num_gpu_card_groups;

	int num_all_dev_groups;

	struct cpu_context *cpu_context;
	struct gpu_context *gpu_core_context;
	struct gpu_context *gpu_card_context;

	float cpu_ratio;
	float gpu_core_ratio;
	float gpu_card_ratio;

};

struct panda_node_context
{	
	keyval_t		*input_keyval_arr;
	keyval_arr_t		*intermediate_keyval_arr_arr_p;
	keyvals_t		*sorted_intermediate_keyvals_arr;
	int 			sorted_keyvals_arr_len;
	
	int num_cpus_groups;
	int num_gpu_core_groups;
	int num_gpu_card_groups;
	
	int num_all_dev_groups;
	
	struct cpu_context 		*cpu_context;
	struct panda_gpu_context 	*gpu_core_context;
	struct gpu_context 		*gpu_card_context;
	
	float cpu_ratio;
	float gpu_core_ratio;
	float gpu_card_ratio;
	
};


//typedef
struct  thread_info_t {

	char *device_name;
	int tid;			//accelerator group id
	//int num_gpu_core_groups;		//
	//int num_cpus;		//num of processors
	char device_type;
	void *d_g_state;	//device context
	void *job_conf;		//job configuration

	void *panda;
	int start_idx;
	int end_idx;

};// thread_info_t;

#define GPU_CORE_ACC		0x01
#define GPU_CARD_ACC		0x05

#define CPU_ACC		0x02
#define CELL_ACC	0x03
#define FPGA_ACC	0x04

typedef int4 cmp_type_t;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Panda  APIs in alphabet order
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C"
void AddPandaTask(job_configuration* job_conf,
						void*		key, 
						void*		val,
						int		keySize, 
						int		valSize);


extern "C"
void AddMapInputRecord2(gpu_context*		spec, 
		   void*		key, 
		   void*		val, 
		   int		keySize, 
		   int		valSize);

extern "C"
void AddMapInputRecord4GPUCore(gpu_context* d_g_state,
						keyval_t *kv_p, int start_row_id, int end_id);

extern "C"
void AddMapInputRecord4GPUCard(gpu_card_context* d_g_state,
						keyval_t *kv_p, int start_row_id, int end_row_id);

extern "C"
void AddReduceInputRecordGPU(gpu_context* d_g_state, 
							 keyvals_t * sorted_intermediate_keyvals_arr, int start_row_id, int end_row_id);

extern "C"
void AddMapInputRecordCPU(cpu_context* d_g_state,
						keyval_t *kv_p, int start_row_id, int end);

extern "C"
void AddReduceInputRecordCPU(cpu_context* d_g_state,
						keyvals_t * kv_p, int start_row_id, int end_row_id);


extern "C"
void GPUCardEmitMapOutput(void *key, 
					  void *val, 
					  int keySize, 
					  int valSize, 
					  gpu_card_context *d_g_state, 
					  int map_task_idx);

extern "C"
void CPUEmitMapOutput(void *key, 
					  void *val, 
					  int keySize, 
					  int valSize, 
					  cpu_context *d_g_state, 
					  int map_task_idx);

extern "C"
void CPUEmitCombinerOutput(void *key, 
						  void *val, 
						  int keySize, 
						  int valSize, 
						  cpu_context *d_g_state, 
						  int map_task_idx);


extern "C"
void CPUEmitReduceOutput (void*		key, 
						void*		val, 
						int		keySize, 
						int		valSize,
		           cpu_context *d_g_state);

extern "C"
void DoDiskLog(const char *str);

extern "C"
void DestroyDGlobalState(gpu_context * d_g_state);

__device__ void GPUEmitCombinerOutput(void*		key, 
						void*		val, 
						int		keySize, 
						int		valSize,
		           	   gpu_context *d_g_state, 
				   int map_task_idx);


__device__ void GPUEmitCombinerOutput(void*             key,
                                                void*           val,
                                                int             keySize,
                                                int             valSize,
                                   		panda_gpu_context *d_g_state,
                                   		int map_task_idx);

__device__ void GPUEmitReduceOuput (void*		key, 
						void*		val, 
						int		keySize, 
						int		valSize,
		           gpu_context *d_g_state);


__device__ void GPUEmitMapOutput(void *key, 
								  void *val, 
								  int keySize, 
								  int valSize, 
								  gpu_context *d_g_state,
								  int map_task_idx);

__device__ void GPUEmitMapOutput(void *key,
                                                                  void *val,
                                                                  int keySize,
                                                                  int valSize,
                                                                  panda_gpu_context *d_g_state,
                                                                  int map_task_idx);


//__device__ int gpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);


extern "C"
void InitGPUMapReduce(gpu_context* d_g_state);

extern "C"
void InitGPUDevice(thread_info_t* thread_info);

extern "C"
void InitCPUDevice(thread_info_t* thread_info);

extern "C"
void PandaMetaScheduler(thread_info_t *thread_info, panda_context *panda);

extern "C"
void *Panda_Map(void *ptr);

//reserve for future usage
extern "C"
void Panda_Shuffle_Merge(gpu_context *d_g_state_0, gpu_context *d_g_state_1);

extern "C"
void PandaShuffleMergeGPU(panda_context *d_g_state_1, gpu_context *d_g_state_0);

void PandaShuffleMergeGPU(panda_node_context *d_g_state_1, panda_gpu_context *d_g_state_0);

void AddReduceTask4GPU(panda_gpu_context* pgc, panda_node_context *pnc, int start_row_id, int end_row_id);

extern "C"
void PandaShuffleMergeCPU(panda_context *d_g_state_1, cpu_context *d_g_state_0);

extern "C"
void PandaShuffleMergeGPUCard(panda_context* panda, gpu_card_context* d_g_state);

extern "C"
void *Panda_Reduce(void *ptr);


void GPUMapPartitionerHost(panda_gpu_context d_g_state, dim3 grids, dim3 blocks);
void RunGPUMapTasksHost(panda_gpu_context pgc, int curIter, int totalIter, dim3 grids, dim3 blocks);
void StartGPUCombiner(panda_gpu_context *pgc);


extern "C"
void Start_Panda_Job(job_configuration*job_conf);

extern "C"
float Smart_Scheduler(thread_info_t *thread_info, panda_context *panda);

extern "C"
int StartGPUCoreMap(gpu_context *d_g_state);

extern "C"
int StartCPUMap(gpu_context *d_g_state);

extern "C"
int StartCPUMap2(thread_info_t* thread_info);

extern "C"
void StartGPUShuffle(gpu_context *d_g_state);

void StartGPUShuffle(panda_gpu_context *d_g_state);

extern "C"
void StartCPUShuffle(cpu_context *d_g_state);

extern "C"
void StartCPUShuffle2(thread_info_t* thread_info);

extern "C"
void StartGPUReduce(gpu_context *d_g_state);

extern "C"
void sort_CPU(gpu_context *d_g_state);

extern "C"
void Shuffle4GPUOutput(gpu_context *d_g_state);

void Shuffle4GPUOutput(panda_gpu_context *d_g_state);

extern "C"
void *RunPandaCPUMapThread(void *ptr);

extern "C"
void *RunPandaCPUCombinerThread(void *ptr);


__global__ 
void RunGPUMapTasks(gpu_context d_g_state, 
					int curIter, 
					int totalIter);

extern "C"
void RunGPUCardMapFunction(gpu_context d_g_state,
						   int curIter, 
						   int totalIter);

extern "C"
int getCPUCoresNum();

extern "C"
int getGPUCoresNum();

extern "C"
panda_gpu_context *CreatePandaGPUContext();

extern "C"
gpu_context *GetDGlobalState();

extern "C"   
cpu_context *CreateCPUContext();

extern "C"
gpu_context *CreateGPUCoreContext();

extern "C"
panda_gpu_context *CreatePandaGPUCoreContext();

extern "C"
gpu_card_context *CreateGPUCardContext();

extern "C"
job_configuration *CreateJobConf();



extern "C"
void MapReduce2(gpu_context *d_g_state);

extern "C"
void FinishMapReduce2(gpu_context* state);


//------------------------------------------------------
//PandaScan.cu
//------------------------------------------------------



//-------------------------------------------------------
//Old PandaLib.cu
//-------------------------------------------------------

#define DEFAULT_DIMBLOCK	256
#define DEFAULT_NUMTASK		1

#define MAP_ONLY		0x01
#define MAP_GROUP		0x02
#define MAP_REDUCE		0x03


typedef struct
{
	
	
} Spec_t;

__device__ void *GetVal(void *vals, int4* interOffsetSizes, int index, int valStartIndex);
__device__ void *GetKey(void *key, int4* interOffsetSizes, int index, int valStartIndex);

extern __shared__ char sbuf[];
#define GET_OUTPUT_BUF(offset) (sbuf + threadIdx.x * 5 * sizeof(int) + offset)
#define GET_VAL_FUNC(vals, index) GetVal(vals, interOffsetSizes, index, valStartIndex)
#define GET_KEY_FUNC(key, index) GetKey(key, interOffsetSizes, index, valStartIndex)

//-------------------------------------------------------
//MarsUtils.cu
//-------------------------------------------------------

__global__ void printData(gpu_context d_g_state );
__global__ void printData2(gpu_context d_g_state );
__global__ void printData3(gpu_context d_g_state);


#ifdef _DEBUG
#define ShowLog(...) do{printf("[#Log#][%s]\t\t",__FUNCTION__);printf(__VA_ARGS__);printf("\n");}while(0)
#else
#define ShowLog(...) //do{printf(__VA_ARGS__);printf("\n");}while(0)
#endif

#ifdef _DISKLOG
#define DoLog2Disk(...) do{ FILE *fptr;	fptr = fopen("panda.log","a"); fprintf(fptr,"[PandaDiskLog]\t\t"); fprintf(fptr,__VA_ARGS__);fprintf(fptr,"\n");fclose(fptr);}while(0)
#else
#define DoLog2Disk(...) 
#endif


#ifdef _ERROR
#define ShowError(...) do{printf("[#Error#][%s]\t\t",__FUNCTION__);printf(__VA_ARGS__);printf("\n");}while(0)
#else
#define ShowError(...)
#endif

#ifdef _WARN
#define ShowWarn(...) do{printf("[#WARN#][%s]\t\t",__FUNCTION__);printf(__VA_ARGS__);printf("\n");}while(0)
#else
#define ShowWarn(...) 
#endif

typedef void (*PrintFunc_t)(void* key, void* val, int keySize, int valSize);
void PrintOutputRecords(Spec_t* spec, int num, PrintFunc_t printFunc);

#endif //__PANDA_H__
