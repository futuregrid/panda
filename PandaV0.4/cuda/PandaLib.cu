/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs.

	Code Name: Panda 
	
	File: PandaLib.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-016

	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef __PANDALIB_CU__
#define __PANDALIB_CU__

#include "Panda.h"
#include "UserAPI.cu"
#include "Global.h"

extern int gCommRank;

//----------------------------------------------
//Get default job configuration
//----------------------------------------------

//3/10/2013
panda_gpu_context *CreatePandaGPUContext(){
	
	panda_gpu_context *pgc = (panda_gpu_context*)malloc(sizeof(panda_gpu_context));
	if (pgc == NULL) exit(-1);
	memset(pgc, 0, sizeof(panda_gpu_context));

	pgc->input_key_vals.d_input_keys_shared_buff = NULL;
	pgc->input_key_vals.d_input_keyval_arr = NULL;
	pgc->input_key_vals.d_input_keyval_pos_arr = NULL;
	pgc->input_key_vals.d_input_vals_shared_buff = NULL;
	pgc->input_key_vals.h_input_keyval_arr = NULL;
	pgc->input_key_vals.num_input_record = 0;

	pgc->intermediate_key_vals.d_intermediate_keys_shared_buff = NULL;
	pgc->intermediate_key_vals.d_intermediate_keyval_arr = NULL;
	pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_len = 0;
	pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p = NULL;
	pgc->intermediate_key_vals.d_intermediate_keyval_pos_arr = NULL;
	pgc->intermediate_key_vals.d_intermediate_keyval_total_count = 0;

	pgc->sorted_key_vals.d_sorted_keyvals_arr_len = 0;
	pgc->reduced_key_vals.d_reduced_keyval_arr_len = 0;

	return pgc;
}//gpu_context


panda_cpu_context *CreatePandaCPUContext(){
	
	panda_cpu_context *pcc = (panda_cpu_context*)malloc(sizeof(panda_cpu_context));
	if (pcc == NULL) exit(-1);
	memset(pcc, 0, sizeof(panda_cpu_context));
	
	pcc->input_key_vals.input_keys_shared_buff = NULL;
	pcc->input_key_vals.input_keyval_arr = NULL;
	pcc->input_key_vals.input_keyval_pos_arr = NULL;
	pcc->input_key_vals.input_vals_shared_buff = NULL;
	pcc->input_key_vals.input_keyval_arr = NULL;
	pcc->input_key_vals.num_input_record = 0;
	
	pcc->intermediate_key_vals.intermediate_keys_shared_buff = NULL;
	pcc->intermediate_key_vals.intermediate_keyval_arr = NULL;
	pcc->intermediate_key_vals.intermediate_keyval_arr_arr_len = 0;
	pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p = NULL;
	pcc->intermediate_key_vals.intermediate_keyval_pos_arr = NULL;
	pcc->intermediate_key_vals.intermediate_keyval_total_count = 0;
	
	pcc->sorted_key_vals.sorted_keyvals_arr_len = 0;
	pcc->reduced_key_vals.reduced_keyval_arr_len = 0;
	return pcc;
	
}//gpu_context


void ExecutePandaSortBucket(panda_node_context *pnc)
{
	  int numBucket = pnc->recv_buckets.savedKeysBuff.size();
	  keyvals_t *sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;
	  char *key_0, *key_1;
	  int keySize_0, keySize_1;
	  char *val_0, val_1;
	  int valSize_0, valSize_1;

	  bool equal;
	  for(int i=0; i<numBucket; i++){
			
		char *keyBuff = pnc->recv_buckets.savedKeysBuff[i];
		char *valBuff = pnc->recv_buckets.savedValsBuff[i];
		int *counts = pnc->recv_buckets.counts[i];

		int *keyPosArray  = pnc->recv_buckets.keyPos[i];
		int *keySizeArray = pnc->recv_buckets.keySize[i];
		int *valPosArray  = pnc->recv_buckets.valPos[i];
		int *valSizeArray = pnc->recv_buckets.valSize[i];

		int maxlen		= counts[0];
		int keyBuffSize	= counts[1];
		int valBuffSize	= counts[2];

		for (int j=0; j<maxlen; j++){
			
			if( keyPosArray[j] + keySizeArray[j] > keyBuffSize ) 
				ShowError("keyPosArray[j]:%d + keySizeArray[j]:%d > keyBuffSize:%d", keyPosArray[j], keySizeArray[j] , keyBuffSize);

			key_0		= keyBuff + keyPosArray[j];
			keySize_0	= keySizeArray[j];

			int k = 0;
			for ( ; k < pnc->sorted_key_vals.sorted_keyvals_arr_len; k++){

				key_1		= (char *)(sorted_intermediate_keyvals_arr[k].key);
				keySize_1	= sorted_intermediate_keyvals_arr[k].keySize;

				if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
					continue;

				val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				int index   = sorted_intermediate_keyvals_arr[k].val_arr_len;
				
				sorted_intermediate_keyvals_arr[k].val_arr_len++;
				sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));
				
				val_0   = valBuff + valPosArray[j];
				valSize_0 = valSizeArray[j];

				sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_0);
				sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_0;
				memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val, val_0, valSize_0);
				break;
			}//for k

			if (k == pnc->sorted_key_vals.sorted_keyvals_arr_len){

			if (pnc->sorted_key_vals.sorted_keyvals_arr_len == 0) sorted_intermediate_keyvals_arr = NULL;

			int index = pnc->sorted_key_vals.sorted_keyvals_arr_len;
			pnc->sorted_key_vals.sorted_keyvals_arr_len++;
			sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*(pnc->sorted_key_vals.sorted_keyvals_arr_len));
			
			keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);

			kvals_p->keySize = keySize_0;
			kvals_p->key = malloc(sizeof(char)*keySize_0);
			memcpy(kvals_p->key, key_0, keySize_0);

			kvals_p->vals = (val_t *)malloc(sizeof(val_t)*1);
			kvals_p->val_arr_len = 1;

			if (valPosArray[j] + valSizeArray[j] > valBuffSize) ShowError("valPosArray[j] + valSizeArray[j] > valBuffSize");

			val_0   = valBuff + valPosArray[j];
			valSize_0 = valSizeArray[j];

			kvals_p->vals[k].valSize = valSize_0;
			kvals_p->vals[k].val = (char *)malloc(sizeof(char)*valSize_0);
			memcpy(kvals_p->vals[k].val, val_0, valSize_0);

			}//k
		}//j
	  }//i
	  pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
}//			
			
void AddReduceTask4CPU(panda_cpu_context* pcc, panda_node_context *pnc, int start_row_id, int end_row_id){
		
    if (end_row_id <= start_row_id){	ShowError("error! end_row_id<=start_row_id");		return;	}
	int len = pnc->sorted_key_vals.sorted_keyvals_arr_len;
	if (len < 0) {	ShowError("error! len<0");		return;	}
	if (len == 0) pcc->sorted_key_vals.sorted_intermediate_keyvals_arr = NULL;
	//realloc()//TODO
	pcc->sorted_key_vals.sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*(len + end_row_id - start_row_id));
	pcc->sorted_key_vals.totalKeySize = pnc->sorted_key_vals.totalKeySize;
	pcc->sorted_key_vals.totalValSize = pnc->sorted_key_vals.totalValSize;
	
	for (int i = len; i< len + end_row_id - start_row_id; i++){
		
		pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].keySize = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr[start_row_id+i-len].keySize;
		pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].key = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr[start_row_id+i-len].key;
		pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].vals = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr[start_row_id+i-len].vals;
		pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].val_arr_len = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr[start_row_id+i-len].val_arr_len;
		
	}//for
	pcc->sorted_key_vals.sorted_keyvals_arr_len = len + end_row_id-start_row_id;
		
}//void AddReduceTask4CPU

void AddReduceTask4GPU(panda_gpu_context* pgc, panda_node_context *pnc, int start_row_id, int end_row_id){

	keyvals_t * sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;
	end_row_id = pnc->sorted_key_vals.sorted_keyvals_arr_len;

	int total_count = 0;
	for(int i=start_row_id;i<end_row_id;i++){
		total_count += sorted_intermediate_keyvals_arr[i].val_arr_len;
	}//for
	
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=start_row_id;i<end_row_id;i++){
		totalKeySize += (sorted_intermediate_keyvals_arr[i].keySize+3)/4*4;
		for (int j=0;j<sorted_intermediate_keyvals_arr[i].val_arr_len;j++)
		totalValSize += (sorted_intermediate_keyvals_arr[i].vals[j].valSize+3)/4*4;
	}//for
		
	checkCudaErrors(cudaMalloc((void **)&pgc->sorted_key_vals.d_sorted_keys_shared_buff, totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&pgc->sorted_key_vals.d_sorted_vals_shared_buff, totalValSize));
	checkCudaErrors(cudaMalloc((void **)&pgc->sorted_key_vals.d_keyval_pos_arr, sizeof(keyval_pos_t)*total_count));
	
	pgc->sorted_key_vals.h_sorted_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	pgc->sorted_key_vals.h_sorted_vals_shared_buff = malloc(sizeof(char)*totalValSize);
	
	char *sorted_keys_shared_buff = (char *)pgc->sorted_key_vals.h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff = (char *)pgc->sorted_key_vals.h_sorted_vals_shared_buff;
	char *keyval_pos_arr = (char *)malloc(sizeof(keyval_pos_t)*total_count);
	
	int sorted_key_arr_len = (end_row_id-start_row_id);
	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	
	//ShowLog("GPU_ID:[%d] total #different intermediate records:%d total records:%d totalKeySize:%d KB totalValSize:%d KB", 
	//	d_g_state->gpu_id, end_row_id - start_row_id, total_count, totalKeySize/1024, totalValSize/1024);

	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*(sorted_key_arr_len));
	memset(pos_arr_4_pos_arr,0,sizeof(int)*sorted_key_arr_len);

	int index = 0;
	int keyPos = 0;
	int valPos = 0;

	for (int i=start_row_id;i<end_row_id;i++){

		keyvals_t* p = (keyvals_t*)&(sorted_intermediate_keyvals_arr[i]);
		memcpy(sorted_keys_shared_buff+keyPos,p->key, p->keySize);

		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = valPos;
			tmp_keyval_pos_arr[index].valSize = p->vals[j].valSize;
			memcpy(sorted_vals_shared_buff + valPos,p->vals[j].val,p->vals[j].valSize);
			valPos += (p->vals[j].valSize+3)/4*4;
			index++;
		}//for
		keyPos += (p->keySize+3)/4*4;
		pos_arr_4_pos_arr[i-start_row_id] = index;
	}//

	pgc->sorted_key_vals.d_sorted_keyvals_arr_len = end_row_id-start_row_id;
	checkCudaErrors(cudaMemcpy(pgc->sorted_key_vals.d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(pgc->sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr),sizeof(int)*sorted_key_arr_len));
	checkCudaErrors(cudaMemcpy(pgc->sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*sorted_key_arr_len,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pgc->sorted_key_vals.d_sorted_keys_shared_buff, sorted_keys_shared_buff, sizeof(char)*totalKeySize,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pgc->sorted_key_vals.d_sorted_vals_shared_buff, sorted_vals_shared_buff, sizeof(char)*totalValSize,cudaMemcpyHostToDevice));

}

__device__ void PandaGPUEmitReduceOutput(void*		key, 
						void*		val, 
						int		keySize, 
						int		valSize,
						panda_gpu_context *pgc){
						
		    keyval_t *p = &(pgc->reduced_key_vals.d_reduced_keyval_arr[TID]);
			p->keySize = keySize;
			p->key = malloc(keySize);
			memcpy(p->key,key,keySize);
			p->valSize = valSize;
			p->val = malloc(valSize);
			memcpy(p->val,val,valSize);

}//__device__ 


__device__ void PandaGPUEmitCombinerOutput(void *key, void *val, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){
			
	keyval_arr_t *kv_arr_p	= pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p[map_task_idx];
	void *shared_buff		= kv_arr_p->shared_buff;
	int shared_buff_len		= *kv_arr_p->shared_buff_len;
	int shared_arr_len		= *kv_arr_p->shared_arr_len;
	int shared_buff_pos		= *kv_arr_p->shared_buff_pos;
		
	int required_mem_len = (shared_buff_pos) + keySize + valSize + sizeof(keyval_pos_t)*(shared_arr_len+1);
	if (required_mem_len> shared_buff_len){

		ShowWarn("Warning! no enough memory in GPU task:%d need:%d KB KeySize:%d ValSize:%d shared_arr_len:%d shared_buff_pos:%d shared_buff_len:%d",
			map_task_idx, required_mem_len/1024,keySize,valSize,shared_arr_len,shared_buff_pos,shared_buff_len);
		
		char *new_buff = (char*)malloc(sizeof(char)*((*kv_arr_p->shared_buff_len)*2));
		if(new_buff==NULL)ShowWarn(" There is not enough memory to allocat!");

		memcpy(new_buff, shared_buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
		memcpy(new_buff + (*kv_arr_p->shared_buff_len)*2 - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len), 
			(char*)shared_buff + (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len),
												sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len));
		
		shared_buff_len = 2*(*kv_arr_p->shared_buff_len);
		(*kv_arr_p->shared_buff_len) = shared_buff_len;	
		
		for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){

		int cur_map_task_idx = kv_arr_p->shared_buddy[idx];  //the buddy relationship won't be changed 
		keyval_arr_t *cur_kv_arr_p = pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p[cur_map_task_idx];
		cur_kv_arr_p->shared_buff = new_buff;
		
		}//for

		free(shared_buff);
		shared_buff = new_buff;
	
	}//if

	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len + 1));
	kv_p->keySize = keySize;
	kv_p->valSize = valSize;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = _COMBINE;			//merged results

	memcpy( (char*)shared_buff + *kv_arr_p->shared_buff_pos, key, keySize);
	kv_p->keyPos = *kv_arr_p->shared_buff_pos;
	*kv_arr_p->shared_buff_pos += (keySize+3)/4*4;

	memcpy( (char*)shared_buff + *kv_arr_p->shared_buff_pos, val, valSize);
	kv_p->valPos = *kv_arr_p->shared_buff_pos;
	*kv_arr_p->shared_buff_pos += (valSize+3)/4*4;
	
	(*kv_arr_p->shared_arr_len)++;
			
}//__device__


__device__ void PandaGPUEmitMapOutput(void *key, void *val, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){
	
	keyval_arr_t *kv_arr_p = pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p[map_task_idx];
	char *buff = (char*)(kv_arr_p->shared_buff);
	
	if (!((*kv_arr_p->shared_buff_pos) + keySize + valSize < (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1))){
		
		ShowWarn("Warning! not enough memory at GPU task:%d *kv_arr_p->shared_arr_len:%d current buff_size:%d KB",
			map_task_idx,*kv_arr_p->shared_arr_len,(*kv_arr_p->shared_buff_len)/1024);
		
		char *new_buff = (char*)malloc(sizeof(char)*((*kv_arr_p->shared_buff_len)*2));
		if(new_buff==NULL){ ShowWarn("Error ! There is not enough memory to allocat!"); return; }
		
		memcpy(new_buff, buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
		memcpy(new_buff + (*kv_arr_p->shared_buff_len)*2 - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len), 
			(char*)buff + (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len),
														sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len));
				
		(*kv_arr_p->shared_buff_len) = 2*(*kv_arr_p->shared_buff_len);
				
		for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){
				
			int cur_map_task_idx = kv_arr_p->shared_buddy[idx];  //the buddy relationship won't be changed 
			keyval_arr_t *cur_kv_arr_p = pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p[cur_map_task_idx];
			cur_kv_arr_p->shared_buff = new_buff;
				
		}//for
		free(buff);//?????
		buff = new_buff;
	}//if
	
	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)buff + *kv_arr_p->shared_buff_len - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1));
	(*kv_arr_p->shared_arr_len)++;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = _MAP;

	kv_p->keyPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((keySize+3)/4)*4;		//alignment 4 bytes for reading and writing
	memcpy((char *)(buff) + kv_p->keyPos,key,keySize);
	kv_p->keySize = keySize;
	
	kv_p->valPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((valSize+3)/4)*4;
	char *val_p = (char *)(buff) + kv_p->valPos;
	memcpy((char *)(buff) + kv_p->valPos, val, valSize);
	kv_p->valSize = valSize;
	(kv_arr_p->arr) = kv_p;
	//kv_arr_p->arr_len++;
	//d_g_state->d_intermediate_keyval_total_count[map_task_idx] = kv_arr_p->arr_len;
}//__device__


__global__ void ExecutePandaGPUMapPartitioner(panda_gpu_context pgc)
{

	//ShowLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d\n",
	//  gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	//if(TID==0) 	ShowWarn("hi 0 -- num_records_per_thread:%d",num_records_per_thread);

	int buddy_arr_len = num_records_per_thread;
	int * int_arr = (int*)malloc((4+buddy_arr_len)*sizeof(int));
	if(int_arr==NULL){ GpuShowError("there is not enough GPU memory\n"); return;}

	int *shared_arr_len = int_arr;
	int *shared_buff_len = int_arr+1;
	int *shared_buff_pos = int_arr+2;
	//int *num_buddy = int_arr+3;
	int *buddy = int_arr+4;
	//if(TID==0) ShowWarn("hi 1");
	(*shared_buff_len) = SHARED_BUFF_LEN;
	(*shared_arr_len) = 0;
	(*shared_buff_pos) = 0;

	char * buff = (char *)malloc(sizeof(char)*(*shared_buff_len));
	keyval_arr_t *kv_arr_t_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*(thread_end_idx-thread_start_idx+STRIDE-1)/STRIDE);
	int index = 0;
	
	for(int idx = thread_start_idx; idx < thread_end_idx; idx += STRIDE){
			buddy[index] = idx;
			index ++;
	}//for
	index = 0;
	//if(TID==0) ShowWarn("hi 2");
	for(int map_task_idx = thread_start_idx; map_task_idx < thread_end_idx; map_task_idx += STRIDE){

		keyval_arr_t *kv_arr_t = (keyval_arr_t *)&(kv_arr_t_arr[index]);
		index++;
		kv_arr_t->shared_buff = buff;
		kv_arr_t->shared_arr_len = shared_arr_len;
		kv_arr_t->shared_buff_len = shared_buff_len;
		kv_arr_t->shared_buff_pos = shared_buff_pos;
		kv_arr_t->shared_buddy = buddy;
		kv_arr_t->shared_buddy_len = buddy_arr_len;
		kv_arr_t->arr = NULL;
		kv_arr_t->arr_len = 0;
		
		//d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx] = kv_arr_t;
		pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[map_task_idx] = kv_arr_t;

	}//for
}

void StartPandaGPUMapPartitioner(panda_gpu_context pgc, dim3 grids, dim3 blocks)
{
   ExecutePandaGPUMapPartitioner<<<grids,blocks>>>(pgc);
}

//TODO change into cudaStream
__global__ void PandaRunGPUMapTasks(panda_gpu_context pgc, int curIter, int totalIter)
{	

	//ShowLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d\n",
	//  gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);
	//ShowLog("num_records_per_thread:%d block_start_idx:%d gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d",num_records_per_thread, block_start_idx, gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;
	if (thread_start_idx + curIter*STRIDE >= thread_end_idx)
		return;
	for(int map_task_idx = thread_start_idx + curIter*STRIDE; map_task_idx < thread_end_idx; map_task_idx += totalIter*STRIDE){
		char *key = (char *)(pgc.input_key_vals.d_input_keys_shared_buff) + pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].keyPos;
		char *val = (char *)(pgc.input_key_vals.d_input_vals_shared_buff) + pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].valPos;
		int valSize = pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].valSize;
		int keySize = pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].keySize;
		//ShowWarn("valSize:%d keySize:%d",valSize,keySize);
		/////////////////////////////////////////////////////////////////////
		gpu_core_map(key, val, keySize, valSize, &pgc, map_task_idx);//
		/////////////////////////////////////////////////////////////////////
	}//for
	keyval_arr_t *kv_arr_p = pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[thread_start_idx];
	//char *shared_buff = (char *)(kv_arr_p->shared_buff);
	//int shared_arr_len = *kv_arr_p->shared_arr_len;
	//int shared_buff_len = *kv_arr_p->shared_buff_len;
	pgc.intermediate_key_vals.d_intermediate_keyval_total_count[thread_start_idx] = *kv_arr_p->shared_arr_len;
	//__syncthreads();
}//GPUMapPartitioner


//void PandaRunGPUMapTasksHost(*pgc, totalIter -1 - iter, totalIter, grids,blocks);
void RunGPUMapTasksHost(panda_gpu_context pgc, int curIter, int totalIter, dim3 grids, dim3 blocks){
	
	PandaRunGPUMapTasks<<<grids,blocks>>>(pgc, totalIter -1 - curIter, totalIter);

}//void


__global__ void GPUCombiner(panda_gpu_context pgc)
{	

	//ShowLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d",
	//  gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	
	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	keyval_arr_t *kv_arr_p = pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[thread_start_idx];

	int *buddy = kv_arr_p->shared_buddy;
	
	//TODO use host function set 
	/*for (int idx=0;idx<kv_arr_p->shared_buddy_len;idx++){
		d_g_state.d_intermediate_keyval_total_count[idx] = 0;
	}*/

	int unmerged_shared_arr_len = *kv_arr_p->shared_arr_len;
	val_t *val_t_arr = (val_t *)malloc(sizeof(val_t)*unmerged_shared_arr_len);
	if (val_t_arr == NULL) GpuShowError("there is no enough memory");

	int num_keyval_pairs_after_combiner = 0;
	for (int i=0; i<unmerged_shared_arr_len;i++){
		
		char *shared_buff = (kv_arr_p->shared_buff);	
		int shared_buff_len = *kv_arr_p->shared_buff_len;

		keyval_pos_t *head_kv_p = (keyval_pos_t *)(shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-i));
		keyval_pos_t *first_kv_p = head_kv_p;

		if (first_kv_p->next_idx != _MAP)
			continue;

		int iKeySize = first_kv_p->keySize;
		char *iKey = shared_buff + first_kv_p->keyPos;
		char *iVal = shared_buff + first_kv_p->valPos;

		if((first_kv_p->keyPos%4!=0)||(first_kv_p->valPos%4!=0)){
			GpuShowError("keyPos or valPos is not aligned with 4 bytes, results could be wrong");
		}

	
		int index = 0;
		first_kv_p = head_kv_p;

		(val_t_arr[index]).valSize = first_kv_p->valSize;
		(val_t_arr[index]).val = (char*)shared_buff + first_kv_p->valPos;

		for (int j=i+1;j<unmerged_shared_arr_len;j++){

			keyval_pos_t *next_kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-j));
			char *jKey = (char *)shared_buff+next_kv_p->keyPos;
			int jKeySize = next_kv_p->keySize;
		
			if (gpu_compare(iKey,iKeySize,jKey,jKeySize)!=0){
				continue;
			}
			index++;
			first_kv_p->next_idx = j;
			first_kv_p = next_kv_p;
			(val_t_arr[index]).valSize = next_kv_p->valSize;
			(val_t_arr[index]).val = (char*)shared_buff + next_kv_p->valPos;
		}

		int valCount = index+1;
		if(valCount>1)
		gpu_combiner(iKey,val_t_arr,iKeySize,(valCount),&pgc,thread_start_idx);
		else{
			first_kv_p->next_idx = _COMBINE;
			first_kv_p->task_idx = thread_start_idx;
		}
		num_keyval_pairs_after_combiner++;
	}//for
	free(val_t_arr);
	pgc.intermediate_key_vals.d_intermediate_keyval_total_count[thread_start_idx] = num_keyval_pairs_after_combiner;
	////////////////////////////////////////////////////////////////////
	__syncthreads();

}//GPUMapPartitioner


void ExecutePandaGPUCombiner(panda_gpu_context * pgc){

	double t1 = PandaTimer();
	//ShowLog("state->num_input_record:%d",pgc->input_key_vals.num_input_record);
	checkCudaErrors(cudaMemset(pgc->intermediate_key_vals.d_intermediate_keyval_total_count,0,pgc->input_key_vals.num_input_record*sizeof(int)));

	int numGPUCores = getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
    dim3 grids(numBlocks, 1);

	GPUCombiner<<<grids,blocks>>>(*pgc);

	cudaThreadSynchronize();
	double t2 = PandaTimer();
	//int *buff;
	//checkCudaErrors(cudaMalloc(
	//	(void **)&buff,100*sizeof(int)));
	//ShowLog("GPU_ID:[%d] GPUCombiner take:%f sec",state->gpu_id, t2-t1);
}

void ExecutePandaCPUCombiner(panda_cpu_context *pcc){
	
	if (pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p == NULL)	{ ShowError("intermediate_keyval_arr_arr_p == NULL"); exit(-1); }
	if (pcc->intermediate_key_vals.intermediate_keyval_arr_arr_len <= 0)	{ ShowError("no any input keys"); exit(-1); }
	if (pcc->num_cpus_cores <= 0) { ShowError("pcc->num_cpus == 0"); exit(-1); }

	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------
	keyval_arr_t *d_keyval_arr_p;
	int *count = NULL;

	int num_threads = pcc->num_cpus_cores;
	int num_records_per_thread = (pcc->input_key_vals.num_input_record + num_threads - 1)/(num_threads);
	int start_row_idx = 0;
	int end_row_idx = 0;

	for (int tid = 0;tid<num_threads;tid++){
		end_row_idx = start_row_idx + num_records_per_thread;
		if (tid < (pcc->input_key_vals.num_input_record % num_threads) )
			end_row_idx++;
		if (end_row_idx > pcc->input_key_vals.num_input_record)
			end_row_idx = pcc->input_key_vals.num_input_record;

		pcc->panda_cpu_task_thread_info[tid].start_row_idx	= start_row_idx;
		pcc->panda_cpu_task_thread_info[tid].end_row_idx	= end_row_idx;
		
		//if(end_row_idx > start_row_idx)
		if (pthread_create(&(pcc->panda_cpu_task_thread[tid]),NULL,RunPandaCPUCombinerThread,(char *)&(pcc->panda_cpu_task_thread_info[tid]))!=0) 
			ShowError("Thread creation failed!");
		start_row_idx = end_row_idx;
	}//for

	for (int tid = 0; tid<num_threads; tid++){
		void *exitstat;
		if (pthread_join(pcc->panda_cpu_task_thread[tid],&exitstat)!=0) ShowError("joining failed");
	}//for

}//void

void *RunPandaCPUCombinerThread(void *ptr){

	panda_cpu_task_info_t *panda_cpu_task_info = (panda_cpu_task_info_t *)ptr;
	panda_cpu_context *pcc	= (panda_cpu_task_info->pcc); 
	panda_node_context *pnc = (panda_cpu_task_info->pnc); 

	int index = 0;
	keyvals_t * merged_keyvals_arr = NULL;
	int merged_key_arr_len = 0;

	int start_idx = panda_cpu_task_info->start_row_idx;
	int end_idx = panda_cpu_task_info->end_row_idx;
	if(start_idx>=end_idx)
		return NULL;

	keyval_arr_t *kv_arr_p	= (keyval_arr_t *)&(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[start_idx]);

	int unmerged_shared_arr_len = *kv_arr_p->shared_arr_len;
    int *shared_buddy			= kv_arr_p->shared_buddy;
    int shared_buddy_len		= kv_arr_p->shared_buddy_len;
	//ShowLog("hi1");
    char *shared_buff = kv_arr_p->shared_buff;
    int shared_buff_len = *kv_arr_p->shared_buff_len;
    int shared_buff_pos = *kv_arr_p->shared_buff_pos;

	val_t *val_t_arr = (val_t *)malloc(sizeof(val_t)*unmerged_shared_arr_len);
	if (val_t_arr == NULL) ShowError("there is no enough memory");
	int num_keyval_pairs_after_combiner = 0;
	int total_intermediate_keyvalue_pairs = 0;

	//ShowLog("hi2");
	for (int i = 0; i < unmerged_shared_arr_len; i++){

		keyval_pos_t *head_kv_p = (keyval_pos_t *)(shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-i));
		keyval_pos_t *first_kv_p = head_kv_p;

		if (first_kv_p->next_idx != _MAP)
			continue;
		//ShowLog("hi3");

		int iKeySize	= first_kv_p->keySize;
		char *iKey		= shared_buff + first_kv_p->keyPos;
		char *iVal		= shared_buff + first_kv_p->valPos;

		if((first_kv_p->keyPos%4!=0)||(first_kv_p->valPos%4!=0)){
			ShowError("keyPos or valPos is not aligned with 4 bytes, results could be wrong");
		}//
	
		int index = 0;
		first_kv_p = head_kv_p;

		(val_t_arr[index]).valSize = first_kv_p->valSize;
		(val_t_arr[index]).val = (char*)shared_buff + first_kv_p->valPos;

		//ShowLog("hi i:%d",i);
		for (int j=i+1;j<unmerged_shared_arr_len;j++){

			keyval_pos_t *next_kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-j));
			char *jKey = (char *)shared_buff+next_kv_p->keyPos;
			int jKeySize = next_kv_p->keySize;
		
			if (cpu_compare(iKey,iKeySize,jKey,jKeySize)!=0){
				continue;
			}

			index++;
			first_kv_p->next_idx = j;
			first_kv_p = next_kv_p;
			(val_t_arr[index]).valSize = next_kv_p->valSize;
			(val_t_arr[index]).val = (char*)shared_buff + next_kv_p->valPos;

		}

		int valCount = index+1;
		total_intermediate_keyvalue_pairs += valCount;
		if(valCount>1)
			panda_cpu_combiner(iKey, val_t_arr, iKeySize, (valCount), pcc, start_idx);
		else{
			first_kv_p->next_idx = _COMBINE;
			first_kv_p->task_idx = start_idx;
		}
		num_keyval_pairs_after_combiner++;
	}//for
	free(val_t_arr);
	pcc->intermediate_key_vals.intermediate_keyval_total_count[start_idx] = num_keyval_pairs_after_combiner;

	/*ShowLog("CPU_GROUP_ID:[%d] Map_Idx:%d  Done:%d Combiner: %d => %d Compress Ratio:%f",
		d_g_state->cpu_group_id, 
		panda_cpu_task_info->start_row_idx,
		panda_cpu_task_info->end_row_idx - panda_cpu_task_info->start_row_idx, 
		total_intermediate_keyvalue_pairs,
		num_keyval_pairs_after_combiner,
		(num_keyval_pairs_after_combiner/(float)total_intermediate_keyvalue_pairs)
		);*/
	return NULL;
}



void StartGPUShuffle(panda_gpu_context * pgc){
	
	double t1 = PandaTimer();
	ExecutePandaGPUSort(pgc);
	double t2 = PandaTimer();
	//ShowLog("GPU_ID:[%d] GPUShuffle take %f sec", state->gpu_id,t2-t1);
	
}//void


//-------------------------------------------------------
//Reducer
//-------------------------------------------------------

__global__ void PandaReducePartitioner(panda_gpu_context pgc)
{
	//ShowError("ReducePartitioner Panda_GPU_Context");
	//int num_records_per_thread = (d_g_state.d_sorted_keyvals_arr_len + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int num_records_per_thread = (pgc.sorted_key_vals.d_sorted_keyvals_arr_len + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	
	if (thread_end_idx > pgc.sorted_key_vals.d_sorted_keyvals_arr_len)
		thread_end_idx = pgc.sorted_key_vals.d_sorted_keyvals_arr_len;

	if (thread_start_idx >= thread_end_idx)
		return;

	int start_idx, end_idx;
	for(int reduce_task_idx=thread_start_idx; reduce_task_idx < thread_end_idx; reduce_task_idx+=STRIDE){

		if (reduce_task_idx==0)
			start_idx = 0;
		else
			//start_idx = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx-1];
			start_idx = pgc.sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx-1];
		end_idx = pgc.sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx];
		val_t *val_t_arr = (val_t*)malloc(sizeof(val_t)*(end_idx-start_idx));
		
		int keySize = pgc.sorted_key_vals.d_keyval_pos_arr[start_idx].keySize;
		int keyPos = pgc.sorted_key_vals.d_keyval_pos_arr[start_idx].keyPos;
		void *key = (char*)pgc.sorted_key_vals.d_sorted_keys_shared_buff+keyPos;
				
		for (int index = start_idx;index<end_idx;index++){
			int valSize = pgc.sorted_key_vals.d_keyval_pos_arr[index].valSize;
			int valPos = pgc.sorted_key_vals.d_keyval_pos_arr[index].valPos;
			val_t_arr[index-start_idx].valSize = valSize;
			val_t_arr[index-start_idx].val = (char*)pgc.sorted_key_vals.d_sorted_vals_shared_buff + valPos;
		}   //for
		if( end_idx - start_idx == 0) GpuShowError("gpu_reduce valCount ==0");
		//else gpu_reduce(key, val_t_arr, keySize, end_idx-start_idx, d_g_state);
		else panda_gpu_reduce(key, val_t_arr, keySize, end_idx-start_idx, pgc);
	}//for
}


void ExecutePandaGPUReduceTasks(panda_gpu_context *pgc)
{
	
	cudaThreadSynchronize(); 
	pgc->reduced_key_vals.d_reduced_keyval_arr_len = pgc->sorted_key_vals.d_sorted_keyvals_arr_len;

	checkCudaErrors(cudaMalloc((void **)&(pgc->reduced_key_vals.d_reduced_keyval_arr), 
		sizeof(keyval_t)*pgc->reduced_key_vals.d_reduced_keyval_arr_len));

	pgc->output_key_vals.totalKeySize = 0;
	pgc->output_key_vals.totalValSize = 0;
	pgc->output_key_vals.h_reduced_keyval_arr_len = pgc->reduced_key_vals.d_reduced_keyval_arr_len;
	pgc->output_key_vals.h_reduced_keyval_arr = (keyval_t*)(malloc(sizeof(keyval_t)*pgc->output_key_vals.h_reduced_keyval_arr_len));

	cudaThreadSynchronize(); 
	int numGPUCores = getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
    dim3 grids(numBlocks, 1);
	
	int total_gpu_threads = (grids.x*grids.y*blocks.x*blocks.y);
	ShowLog("reduce len:%d intermediate len:%d output len:%d sorted keySize%d: sorted valSize:%d",
		pgc->reduced_key_vals.d_reduced_keyval_arr_len, 
		pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_len,
		pgc->output_key_vals.h_reduced_keyval_arr_len,
		pgc->sorted_key_vals.totalKeySize, 
		pgc->sorted_key_vals.totalValSize);

	PandaReducePartitioner<<<grids,blocks>>>(*pgc);

	checkCudaErrors(cudaMemcpy(pgc->output_key_vals.h_reduced_keyval_arr,
		pgc->reduced_key_vals.d_reduced_keyval_arr,
		sizeof(keyval_t)*pgc->reduced_key_vals.d_reduced_keyval_arr_len,
		cudaMemcpyDeviceToHost));

	for (int i = 0; i<pgc->reduced_key_vals.d_reduced_keyval_arr_len; i++){
		//ShowLog("keySize:%d\n",pgc->output_key_vals.h_reduced_keyval_arr[i].keySize);
		//ShowLog("valSize:%d\n",pgc->output_key_vals.h_reduced_keyval_arr[i].valSize);
		pgc->output_key_vals.totalKeySize += (pgc->output_key_vals.h_reduced_keyval_arr[i].keySize+3)/4*4;
		pgc->output_key_vals.totalValSize += (pgc->output_key_vals.h_reduced_keyval_arr[i].valSize+3)/4*4;
	}//for
	
	ShowLog("Output total keySize:%d valSize:%d\n",pgc->output_key_vals.totalKeySize,pgc->output_key_vals.totalValSize);

	pgc->output_key_vals.h_KeyBuff = malloc(sizeof(char)*pgc->output_key_vals.totalKeySize);
	pgc->output_key_vals.h_ValBuff = malloc(sizeof(char)*pgc->output_key_vals.totalValSize);

	checkCudaErrors(cudaMalloc(&(pgc->output_key_vals.d_KeyBuff), sizeof(char)*pgc->output_key_vals.totalKeySize ));
	checkCudaErrors(cudaMalloc(&(pgc->output_key_vals.d_ValBuff), sizeof(char)*pgc->output_key_vals.totalValSize ));

	copyDataFromDevice2Host4Reduce<<<grids,blocks>>>(*pgc);

	checkCudaErrors(cudaMemcpy(
			pgc->output_key_vals.h_KeyBuff,
			pgc->output_key_vals.d_KeyBuff,
			pgc->output_key_vals.totalKeySize,
		cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(
		pgc->output_key_vals.h_ValBuff,
		pgc->output_key_vals.d_ValBuff,
		pgc->output_key_vals.totalValSize,
		cudaMemcpyDeviceToHost));

	int val_pos, key_pos;
	val_pos = key_pos = 0;
	void *val, *key;

	for (int i = 0; i<pgc->output_key_vals.h_reduced_keyval_arr_len; i++){
		
		val = (char *)pgc->output_key_vals.h_ValBuff + val_pos;
		key = (char *)pgc->output_key_vals.h_KeyBuff + key_pos;
		pgc->output_key_vals.h_reduced_keyval_arr[i].key = key;
		pgc->output_key_vals.h_reduced_keyval_arr[i].val = val;
		ShowLog("key:%s val:%d\n",key,*(int*)val);

		val_pos += (pgc->output_key_vals.h_reduced_keyval_arr[i].valSize+3)/4*4;
		key_pos += (pgc->output_key_vals.h_reduced_keyval_arr[i].keySize+3)/4*4;

	}//for

	//ShowLog("Output: Size:%d keySize:%d",pgc->reduced_key_vals.d_reduced_keyval_arr_len,pgc->reduced_key_vals.h_reduced_keyval_arr[0].keySize);

	//TODO
	//cudaThreadSynchronize(); 

}//void

	
void* ExecutePandaCPUMapThread(void * ptr)
{

	panda_cpu_task_info_t *panda_cpu_task_info = (panda_cpu_task_info_t *)ptr;
	panda_cpu_context  *pcc = (panda_cpu_context *) (panda_cpu_task_info->pcc);
	panda_node_context *pnc = (panda_node_context *)(panda_cpu_task_info->pnc);
	
	int start_row_idx	=	panda_cpu_task_info->start_row_idx;
	int end_row_idx		=	panda_cpu_task_info->end_row_idx;

	if(end_row_idx<=start_row_idx) 	return NULL;
	
	char *buff		=	(char *)malloc(sizeof(char)*CPU_SHARED_BUFF_SIZE);
	int *int_arr	=	(int *)malloc(sizeof(int)*(end_row_idx - start_row_idx + 3));
	int *buddy		=	int_arr+3;
	
	int buddy_len	=	end_row_idx	- start_row_idx;
	
	for (int i=0;i<buddy_len;i++){
		buddy [i]	=	i + start_row_idx;
	}//for
	
	for (int map_idx = start_row_idx; map_idx < end_row_idx; map_idx++){

		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff)		= buff;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_len) = int_arr;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_pos) = int_arr+1;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_arr_len)	= int_arr+2;
		
		*(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_len)	= CPU_SHARED_BUFF_SIZE;
		*(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_pos)	= 0;
		*(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_arr_len)		= 0;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buddy)		= buddy;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buddy_len)	= buddy_len;

	}//for

	for (int map_idx = panda_cpu_task_info->start_row_idx; map_idx < panda_cpu_task_info->end_row_idx; map_idx++){

		keyval_t *kv_p = (keyval_t *)(&(pcc->input_key_vals.input_keyval_arr[map_idx]));
		panda_cpu_map(kv_p->key,kv_p->val,kv_p->keySize,kv_p->valSize,pcc,map_idx);

	}//for
	
	//ShowLog("CPU_GROUP_ID:[%d] Done :%d tasks",d_g_state->cpu_group_id, panda_cpu_task_info->end_row_idx - panda_cpu_task_info->start_row_idx);
	return NULL;
}//int 


void PandaEmitCPUMapOutput(void *key, void *val, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){
	
	if(map_task_idx >= pcc->input_key_vals.num_input_record) {	ShowError("error ! map_task_idx >= d_g_state->num_input_record");		return;	}
	keyval_arr_t *kv_arr_p = &(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_task_idx]);

	char *buff = (char*)(kv_arr_p->shared_buff);
	
	if (!((*kv_arr_p->shared_buff_pos) + keySize + valSize < (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1))){
		ShowWarn("Warning! not enough memory at CPU task:%d *kv_arr_p->shared_arr_len:%d current buff_size:%d KB",
			map_task_idx,*kv_arr_p->shared_arr_len,(*kv_arr_p->shared_buff_len)/1024);

		char *new_buff = (char*)malloc(sizeof(char)*((*kv_arr_p->shared_buff_len)*2));
		if(new_buff==NULL){ ShowError("Error ! There is not enough memory to allocat!"); return; }

		memcpy(new_buff, buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
		int blockSize = sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len);
		memcpy(new_buff + (*kv_arr_p->shared_buff_len)*2 - blockSize, 
			(char*)buff + (*kv_arr_p->shared_buff_len) - blockSize,
														blockSize);
		
		(*kv_arr_p->shared_buff_len) = 2*(*kv_arr_p->shared_buff_len);

		for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){
			int cur_map_task_idx = kv_arr_p->shared_buddy[idx];  //the buddy relationship won't be changed
			keyval_arr_t *cur_kv_arr_p = &(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[cur_map_task_idx]);
			cur_kv_arr_p->shared_buff = new_buff;
		}//for
		free(buff);//
		buff = new_buff;
	}//if
	
	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)buff + *kv_arr_p->shared_buff_len - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1));
	(*kv_arr_p->shared_arr_len)++;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = _MAP;
	
	kv_p->keyPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((keySize+3)/4)*4;		//alignment 4 bytes for reading and writing
	memcpy((char *)(buff) + kv_p->keyPos, key, keySize);
	kv_p->keySize = keySize;
	
	kv_p->valPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((valSize+3)/4)*4;
	char *val_p = (char *)(buff) + kv_p->valPos;
	memcpy((char *)(buff) + kv_p->valPos, val, valSize);
	kv_p->valSize = valSize;
	(kv_arr_p->arr) = kv_p;

}//__device__



#endif //__PANDALIB_CU__