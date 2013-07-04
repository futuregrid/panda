/*
Copyright 2012 The Trustees of Indiana University.  All rights reserved.
CGL MapReduce Framework on GPUs and CPUs

Code Name: Panda

File: PandaSort.cu
First Version:		2012-07-01 V0.1
Current Version:	2012-09-01 V0.3
Last Updates:		2012-09-16

Developer: Hui Li (lihui@indiana.edu)

This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
*/

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <math.h>

#include <cuda_runtime.h>

#ifndef _PANDASORT_CU_
#define _PANDASORT_CU_

#include "Panda.h"
#include "CmeansAPI.h"

//void initialize(cmp_type_t *d_data, int rLen, cmp_type_t value)
//{
//	cudaThreadSynchronize();
//}//void initialize


__global__ void copyDataFromDevice2Host1(panda_gpu_context pgc)
{	

	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;

	if(thread_end_idx>pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	int begin=0;
	int end=0;
	for (int i=0; i<thread_start_idx; i++){
		begin += pgc.intermediate_key_vals.d_intermediate_keyval_total_count[i];
	}//for
	end = begin + pgc.intermediate_key_vals.d_intermediate_keyval_total_count[thread_start_idx];

	int start_idx = 0;
	//bool local_combiner = d_g_state.local_combiner;
	bool local_combiner = true;

	for(int i=begin;i<end;i++){
		keyval_t * p1 = &(pgc.intermediate_key_vals.d_intermediate_keyval_arr[i]);
		keyval_pos_t * p2 = NULL;
		keyval_arr_t *kv_arr_p = pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[thread_start_idx];

		char *shared_buff = (char *)(kv_arr_p->shared_buff);
		int shared_arr_len = *kv_arr_p->shared_arr_len;
		int shared_buff_len = *kv_arr_p->shared_buff_len;

		for (int idx = start_idx; idx<(shared_arr_len); idx++){
			p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - idx ));

			if ( local_combiner && p2->next_idx != _COMBINE ){
				continue;
			}//if

			start_idx = idx+1;

			p1->keySize = p2->keySize;
			p1->valSize = p2->valSize;
			p1->task_idx = i;
			p2->task_idx = i;
			break;
		}//for
	}//for
}

__global__ void copyDataFromDevice2Host2(panda_gpu_context pgc)
{	

	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;

	if(thread_end_idx>pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	int begin, end;
	begin=end=0;
	for (int i=0; i<thread_start_idx; i++) 	
		begin = begin + pgc.intermediate_key_vals.d_intermediate_keyval_total_count[i];
	end = begin + pgc.intermediate_key_vals.d_intermediate_keyval_total_count[thread_start_idx];

	keyval_arr_t *kv_arr_p = pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[thread_start_idx];
	char *shared_buff = (char *)(kv_arr_p->shared_buff);
	int shared_arr_len = *kv_arr_p->shared_arr_len;
	int shared_buff_len = *kv_arr_p->shared_buff_len;

	int val_pos, key_pos;
	char *val_p,*key_p;
	int counter = 0;
	//bool local_combiner = d_g_state.local_combiner;
	bool local_combiner = true;

	for(int local_idx = 0; local_idx<(shared_arr_len); local_idx++){

	keyval_pos_t *p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - local_idx ));
	if (local_combiner && p2->next_idx != _COMBINE)		continue;
	//	if (p2->task_idx != i) 		continue;
	int global_idx = p2->task_idx;
	val_pos = pgc.intermediate_key_vals.d_intermediate_keyval_pos_arr[global_idx].valPos;
	key_pos = pgc.intermediate_key_vals.d_intermediate_keyval_pos_arr[global_idx].keyPos;

	val_p = (char*)(pgc.intermediate_key_vals.d_intermediate_vals_shared_buff)+val_pos;
	key_p = (char*)(pgc.intermediate_key_vals.d_intermediate_keys_shared_buff)+key_pos;

	memcpy(key_p, shared_buff + p2->keyPos, p2->keySize);
	memcpy(val_p, shared_buff + p2->valPos, p2->valSize);

	counter++;
	}
	//if(counter!=end-begin)
	//	ShowWarn("counter!=end-begin counter:%d end-begin:%d",counter,end-begin);
	
	free(shared_buff);

}//__global__	

__global__ void copyDataFromDevice2Host4Reduce(panda_gpu_context pgc)
{	

	int num_records_per_thread = (pgc.reduced_key_vals.d_reduced_keyval_arr_len
		+ (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);

	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ (threadIdx.y*blockDim.x + threadIdx.x)*num_records_per_thread;

	int thread_end_idx = thread_start_idx + num_records_per_thread;
	if(thread_end_idx> pgc.reduced_key_vals.d_reduced_keyval_arr_len)
		thread_end_idx = pgc.reduced_key_vals.d_reduced_keyval_arr_len;

	if (thread_start_idx >= thread_end_idx)
		return;

	int val_pos, key_pos;
	for (int i=0; i<thread_start_idx; i++){
		val_pos += (pgc.reduced_key_vals.d_reduced_keyval_arr[i].valSize+3)/4*4;
		key_pos += (pgc.reduced_key_vals.d_reduced_keyval_arr[i].keySize+3)/4*4;
	}//for

	for (int i = thread_start_idx; i < thread_end_idx;i++){
		memcpy( (char *)(pgc.output_key_vals.d_KeyBuff) + key_pos,
			(char *)(pgc.reduced_key_vals.d_reduced_keyval_arr[i].key), pgc.reduced_key_vals.d_reduced_keyval_arr[i].keySize);
		key_pos += pgc.reduced_key_vals.d_reduced_keyval_arr[i].keySize;
		memcpy( (char *)(pgc.output_key_vals.d_ValBuff) + val_pos,
			(char *)(pgc.reduced_key_vals.d_reduced_keyval_arr[i].val), pgc.reduced_key_vals.d_reduced_keyval_arr[i].valSize);
		val_pos += pgc.reduced_key_vals.d_reduced_keyval_arr[i].valSize;
	}//for

}//__global__	


void PandaExecuteSortOnGPUCard(panda_gpu_card_context *pgcc, panda_node_context *pnc){
	
	int index = 0;
	int merged_key_arr_len = 0;
	keyvals_t * merged_keyvals_arr = NULL;
	
	int start_idx = 0;
	int end_idx = 0;
	int total_count = 0;
	for (int i=0; i < pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_len; i++){
		total_count += pgcc->intermediate_key_vals.intermediate_keyval_total_count[i];
	}//for

	int keyvals_arr_len = pnc->sorted_key_vals.sorted_keyval_arr_max_len;
	//pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*keyvals_arr_len);
	keyvals_t * sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;

	int sorted_key_arr_len = 0;
	int num_threads = 1;
	int num_records_per_thread = pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_len;

	for (int tid = 0;tid<num_threads;tid++){
	
		end_idx = start_idx + num_records_per_thread;
		if (tid < (pgcc->input_key_vals.num_input_record % num_threads) )
			end_idx++;
			
		if (end_idx > pgcc->input_key_vals.num_input_record)
			end_idx = pgcc->input_key_vals.num_input_record;
		
		if (end_idx<=start_idx) continue;
		keyval_arr_t *kv_arr_p = (keyval_arr_t *)&(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[start_idx]);

		int shared_arr_len = *kv_arr_p->shared_arr_len;
		int *shared_buddy = kv_arr_p->shared_buddy;
		int shared_buddy_len = kv_arr_p->shared_buddy_len;

		char *shared_buff = kv_arr_p->shared_buff;
		int shared_buff_len = *kv_arr_p->shared_buff_len;
		int shared_buff_pos = *kv_arr_p->shared_buff_pos;

		int val_pos, key_pos;
		char *val_p,*key_p;
		int counter = 0;
		//bool local_combiner = pnc->local_combiner;
		//TODO
		bool local_combiner = true;

		for(int local_idx = 0; local_idx<(shared_arr_len); local_idx++){

			keyval_pos_t *p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - local_idx ));
			if (local_combiner && p2->next_idx != _COMBINE)		continue;
		
			char *key_i = shared_buff + p2->keyPos;
			char *val_i = shared_buff + p2->valPos;

			int keySize_i = p2->keySize;
			int valSize_i = p2->valSize;
		
			int k = 0;
			for (; k<sorted_key_arr_len; k++){
				char *key_k = (char *)(sorted_intermediate_keyvals_arr[k].key);
				int keySize_k = sorted_intermediate_keyvals_arr[k].keySize;

				if ( panda_cpu_compare(key_i, keySize_i, key_k, keySize_k) != 0 )
					continue;

				//found the match
				val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				sorted_intermediate_keyvals_arr[k].val_arr_len++;
				sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));

				int index = sorted_intermediate_keyvals_arr[k].val_arr_len - 1;
				sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_i;
				sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val,val_i,valSize_i);
				break;

			}//for
			
			if (k == sorted_key_arr_len){

				sorted_key_arr_len++;
				if (sorted_key_arr_len >= keyvals_arr_len){

					keyvals_arr_len*=2;
					keyvals_t* new_sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*keyvals_arr_len);
					memcpy(new_sorted_intermediate_keyvals_arr, sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*keyvals_arr_len/2);
					sorted_intermediate_keyvals_arr=new_sorted_intermediate_keyvals_arr;

				}//if

				//sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*sorted_key_arr_len);
				int index = sorted_key_arr_len-1;
				keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);
				kvals_p->keySize = keySize_i;

				kvals_p->key = malloc(sizeof(char)*keySize_i);
				memcpy(kvals_p->key, key_i, keySize_i);

				kvals_p->vals = (val_t *)malloc(sizeof(val_t));
				kvals_p->val_arr_len = 1;

				kvals_p->vals[0].valSize = valSize_i;
				kvals_p->vals[0].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(kvals_p->vals[0].val,val_i, valSize_i);

				pnc->sorted_key_vals.sorted_keyval_arr_max_len = keyvals_arr_len;

			}//if
			
		}
	
		free(shared_buff);
		start_idx = end_idx;
		pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
	}
	pnc->sorted_key_vals.sorted_keyvals_arr_len = sorted_key_arr_len;
	
}//void

void PandaExecuteSortOnCPU(panda_cpu_context *pcc, panda_node_context *pnc){

	int index = 0;
	int merged_key_arr_len = 0;
	keyvals_t * merged_keyvals_arr = NULL;

	int num_threads = pcc->num_cpus_cores;
	int num_records_per_thread = (pcc->input_key_vals.num_input_record)/(num_threads);
	
	int start_idx = 0;
	int end_idx = 0;
	
	int total_count = 0;
	for (int i=0; i< pcc->input_key_vals.num_input_record; i++){
		total_count += pcc->intermediate_key_vals.intermediate_keyval_total_count[i];
	}//for

	int keyvals_arr_len = pnc->sorted_key_vals.sorted_keyval_arr_max_len;
	//pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*keyvals_arr_len);
	keyvals_t * sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;
				
	int sorted_key_arr_len = 0;

	for (int tid = 0;tid<num_threads;tid++){
	
		end_idx = start_idx + num_records_per_thread;
		if (tid < (pcc->input_key_vals.num_input_record % num_threads) )
			end_idx++;
			
		if (end_idx > pcc->input_key_vals.num_input_record)
			end_idx = pcc->input_key_vals.num_input_record;
		
		if (end_idx<=start_idx) continue;

		keyval_arr_t *kv_arr_p = (keyval_arr_t *)&(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[start_idx]);

		int shared_arr_len = *kv_arr_p->shared_arr_len;
		int *shared_buddy = kv_arr_p->shared_buddy;
		int shared_buddy_len = kv_arr_p->shared_buddy_len;

		char *shared_buff = kv_arr_p->shared_buff;
		int shared_buff_len = *kv_arr_p->shared_buff_len;
		int shared_buff_pos = *kv_arr_p->shared_buff_pos;

		int val_pos, key_pos;
		char *val_p,*key_p;
		int counter = 0;
		//bool local_combiner = pnc->local_combiner;
		//TODO
		bool local_combiner = true;

		for(int local_idx = 0; local_idx<(shared_arr_len); local_idx++){

			keyval_pos_t *p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - local_idx ));
			if (local_combiner && p2->next_idx != _COMBINE)		continue;
		
			char *key_i = shared_buff + p2->keyPos;
			char *val_i = shared_buff + p2->valPos;

			int keySize_i = p2->keySize;
			int valSize_i = p2->valSize;
		
			int k = 0;
			for (; k<sorted_key_arr_len; k++){
				char *key_k = (char *)(sorted_intermediate_keyvals_arr[k].key);
				int keySize_k = sorted_intermediate_keyvals_arr[k].keySize;

				if ( panda_cpu_compare(key_i, keySize_i, key_k, keySize_k) != 0 )
					continue;

				//found the match
				val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				sorted_intermediate_keyvals_arr[k].val_arr_len++;
				sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));

				int index = sorted_intermediate_keyvals_arr[k].val_arr_len - 1;
				sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_i;
				sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val,val_i,valSize_i);
				break;

			}//for
			
			if (k == sorted_key_arr_len){
				sorted_key_arr_len++;
				if (sorted_key_arr_len >= keyvals_arr_len){

					keyvals_arr_len*=2;
					keyvals_t* new_sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*keyvals_arr_len);
					memcpy(new_sorted_intermediate_keyvals_arr, sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*keyvals_arr_len/2);
					sorted_intermediate_keyvals_arr=new_sorted_intermediate_keyvals_arr;

				}//if

				//sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*sorted_key_arr_len);
				int index = sorted_key_arr_len-1;
				keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);
				kvals_p->keySize = keySize_i;

				kvals_p->key = malloc(sizeof(char)*keySize_i);
				memcpy(kvals_p->key, key_i, keySize_i);

				kvals_p->vals = (val_t *)malloc(sizeof(val_t));
				kvals_p->val_arr_len = 1;

				kvals_p->vals[0].valSize = valSize_i;
				kvals_p->vals[0].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(kvals_p->vals[0].val,val_i, valSize_i);
				pnc->sorted_key_vals.sorted_keyval_arr_max_len = keyvals_arr_len;

			}//if
		}
	
		free(shared_buff);
		start_idx = end_idx;
		pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
	}
	pnc->sorted_key_vals.sorted_keyvals_arr_len = sorted_key_arr_len;

}


void PandaExecuteSortOnGPU(panda_gpu_context* pgc){

	cudaThreadSynchronize();
	int *count_arr = (int *)malloc(sizeof(int) * pgc->input_key_vals.num_input_record);
	checkCudaErrors(cudaMemcpy(count_arr, pgc->intermediate_key_vals.d_intermediate_keyval_total_count, 
		sizeof(int)*pgc->input_key_vals.num_input_record, cudaMemcpyDeviceToHost));

	int total_count = 0;
	for(int i=0;i<pgc->input_key_vals.num_input_record;i++){
		total_count += count_arr[i];
	}//for
	free(count_arr);

	ShowLog("Total Count of Intermediate Records:%d",total_count);
	checkCudaErrors(cudaMalloc((void **)&(pgc->intermediate_key_vals.d_intermediate_keyval_arr),sizeof(keyval_t)*total_count));

	int num_mappers = 1;
	int num_blocks = (num_mappers + (NUM_THREADS)-1)/(NUM_THREADS);
	//int num_blocks = (pgc->input_key_vals->num_mappers + (NUM_THREADS)-1)/(NUM_THREADS);
	int numGPUCores = getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
	dim3 grids(numBlocks, 1);

	copyDataFromDevice2Host1<<<grids,blocks>>>(*pgc);
	cudaThreadSynchronize();

	//TODO intermediate keyval_arr use pos_arr
	keyval_t * h_keyval_arr = (keyval_t *)malloc(sizeof(keyval_t)*total_count);
	checkCudaErrors(cudaMemcpy(h_keyval_arr, pgc->intermediate_key_vals.d_intermediate_keyval_arr, 
		sizeof(keyval_t)*total_count, cudaMemcpyDeviceToHost));

	pgc->intermediate_key_vals.h_intermediate_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	keyval_pos_t *h_intermediate_keyvals_pos_arr = pgc->intermediate_key_vals.h_intermediate_keyval_pos_arr;

	int totalKeySize = 0;
	int totalValSize = 0;

	for (int i=0;i<total_count;i++){
		h_intermediate_keyvals_pos_arr[i].valPos= totalValSize;
		h_intermediate_keyvals_pos_arr[i].keyPos = totalKeySize;

		h_intermediate_keyvals_pos_arr[i].keySize = h_keyval_arr[i].keySize;
		h_intermediate_keyvals_pos_arr[i].valSize = h_keyval_arr[i].valSize;

		totalKeySize += (h_keyval_arr[i].keySize+3)/4*4;
		totalValSize += (h_keyval_arr[i].valSize+3)/4*4;
	}//for

	if ((totalValSize<=0)||(totalKeySize<=0)){
		ShowError("(totalValSize<=0)||(totalKeySize<=0)  Exit!");
		exit(0);
	}//if

	pgc->sorted_key_vals.totalValSize = totalValSize;
	pgc->sorted_key_vals.totalKeySize = totalKeySize;

	ShowLog("allocate memory of totalKeySize:%f KB totalValSize:%f KB, number of intermediate records:%d ", (double)(totalKeySize)/1024.0, (double)totalValSize/1024.0, total_count);
	checkCudaErrors(cudaMalloc((void **)&pgc->intermediate_key_vals.d_intermediate_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&pgc->intermediate_key_vals.d_intermediate_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&pgc->intermediate_key_vals.d_intermediate_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
	checkCudaErrors(cudaMemcpy(pgc->intermediate_key_vals.d_intermediate_keyval_pos_arr, h_intermediate_keyvals_pos_arr, sizeof(keyval_pos_t)*total_count, cudaMemcpyHostToDevice));

	cudaThreadSynchronize();
	copyDataFromDevice2Host2<<<grids,blocks>>>(*pgc);
	cudaThreadSynchronize();

	pgc->intermediate_key_vals.h_intermediate_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	pgc->intermediate_key_vals.h_intermediate_vals_shared_buff = malloc(sizeof(char)*totalValSize);

	checkCudaErrors(cudaMemcpy(pgc->intermediate_key_vals.h_intermediate_keys_shared_buff,pgc->intermediate_key_vals.d_intermediate_keys_shared_buff,sizeof(char)*totalKeySize,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pgc->intermediate_key_vals.h_intermediate_vals_shared_buff,pgc->intermediate_key_vals.d_intermediate_vals_shared_buff,sizeof(char)*totalValSize,cudaMemcpyDeviceToHost));

	//////////////////////////////////////////////
	checkCudaErrors(cudaMalloc((void **)&pgc->sorted_key_vals.d_sorted_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&pgc->sorted_key_vals.d_sorted_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&pgc->sorted_key_vals.d_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));

	pgc->sorted_key_vals.h_sorted_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	pgc->sorted_key_vals.h_sorted_vals_shared_buff = malloc(sizeof(char)*totalValSize);

	char *sorted_keys_shared_buff = (char *)pgc->sorted_key_vals.h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff = (char *)pgc->sorted_key_vals.h_sorted_vals_shared_buff;

	char *intermediate_key_shared_buff = (char *)pgc->intermediate_key_vals.h_intermediate_keys_shared_buff;
	char *intermediate_val_shared_buff = (char *)pgc->intermediate_key_vals.h_intermediate_vals_shared_buff;

	memcpy(sorted_keys_shared_buff, intermediate_key_shared_buff, totalKeySize);
	memcpy(sorted_vals_shared_buff, intermediate_val_shared_buff, totalValSize);

	int sorted_key_arr_len = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//transfer the d_sorted_keyval_pos_arr to h_sorted_keyval_pos_arr

	sorted_keyval_pos_t * h_sorted_keyval_pos_arr = NULL;
	for (int i=0; i<total_count; i++){
		int iKeySize = h_intermediate_keyvals_pos_arr[i].keySize;

		int j = 0;
		for (; j<sorted_key_arr_len; j++){

			int jKeySize = h_sorted_keyval_pos_arr[j].keySize;
			char *key_i = (char *)(intermediate_key_shared_buff + h_intermediate_keyvals_pos_arr[i].keyPos);
			char *key_j = (char *)(sorted_keys_shared_buff + h_sorted_keyval_pos_arr[j].keyPos);
			if (panda_cpu_compare(key_i,iKeySize,key_j,jKeySize)!=0)
				continue;

			//found the match
			int arr_len = h_sorted_keyval_pos_arr[j].val_arr_len;
			h_sorted_keyval_pos_arr[j].val_pos_arr = (val_pos_t *)realloc(h_sorted_keyval_pos_arr[j].val_pos_arr, sizeof(val_pos_t)*(arr_len+1));
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
			h_sorted_keyval_pos_arr[j].val_arr_len ++;
			break;
		}//for

		if(j==sorted_key_arr_len){
			sorted_key_arr_len++;
			h_sorted_keyval_pos_arr = (sorted_keyval_pos_t *)realloc(h_sorted_keyval_pos_arr,sorted_key_arr_len*sizeof(sorted_keyval_pos_t));
			sorted_keyval_pos_t *p = &(h_sorted_keyval_pos_arr[sorted_key_arr_len - 1]);
			p->keySize = iKeySize;
			p->keyPos = h_intermediate_keyvals_pos_arr[i].keyPos;

			p->val_arr_len = 1;
			p->val_pos_arr = (val_pos_t*)malloc(sizeof(val_pos_t));
			p->val_pos_arr[0].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			p->val_pos_arr[0].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
		}//if

	}

	pgc->sorted_key_vals.h_sorted_keyval_pos_arr	= h_sorted_keyval_pos_arr;
	pgc->sorted_key_vals.d_sorted_keyvals_arr_len	= sorted_key_arr_len;

	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	//ShowLog("GPU_ID:[%d] #input_records:%d #intermediate_records:%lu #different_intermediate_records:%d totalKeySize:%d KB totalValSize:%d KB", 
	//	d_g_state->gpu_id, d_g_state->num_input_record, total_count, sorted_key_arr_len,totalKeySize/1024,totalValSize/1024);

	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*sorted_key_arr_len);
	memset(pos_arr_4_pos_arr,0,sizeof(int)*sorted_key_arr_len);

	int	index = 0;
	for (int i=0;i<sorted_key_arr_len;i++){
		sorted_keyval_pos_t *p = (sorted_keyval_pos_t *)&(h_sorted_keyval_pos_arr[i]);

		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = p->keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = p->val_pos_arr[j].valPos;
			tmp_keyval_pos_arr[index].valSize = p->val_pos_arr[j].valSize;
			index++;
		}//for
		pos_arr_4_pos_arr[i] = index;

	}

	checkCudaErrors(cudaMemcpy(pgc->sorted_key_vals.d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice));
	pgc->sorted_key_vals.d_sorted_keyvals_arr_len = sorted_key_arr_len;
	checkCudaErrors(cudaMalloc((void**)&pgc->sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr,sizeof(int)*sorted_key_arr_len));
	checkCudaErrors(cudaMemcpy(pgc->sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*sorted_key_arr_len,cudaMemcpyHostToDevice));

}

#if 0  //preserved for reference

void PandaShuffleMergeCPU(panda_context *d_g_state_0, cpu_context *d_g_state_1){

	ShowLog("PandaShuffleMergeCPU CPU_GROUP_ID:[%d]", d_g_state_1->cpu_group_id);
	keyvals_t * panda_sorted_intermediate_keyvals_arr = d_g_state_0->sorted_intermediate_keyvals_arr;
	keyvals_t * cpu_sorted_intermediate_keyvals_arr = d_g_state_1->sorted_intermediate_keyvals_arr;

	void *key_0, *key_1;
	int keySize_0, keySize_1;
	bool equal;	

	for (int i=0; i<d_g_state_1->sorted_keyvals_arr_len; i++){
		key_1 = cpu_sorted_intermediate_keyvals_arr[i].key;
		keySize_1 = cpu_sorted_intermediate_keyvals_arr[i].keySize;

		int j;
		for (j=0; j<d_g_state_0->sorted_keyvals_arr_len; j++){
			key_0 = panda_sorted_intermediate_keyvals_arr[j].key;
			keySize_0 = panda_sorted_intermediate_keyvals_arr[j].keySize;
			
			if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;

			//copy values from cpu_contex to panda context
			int val_arr_len_1 = cpu_sorted_intermediate_keyvals_arr[i].val_arr_len;
			int index = panda_sorted_intermediate_keyvals_arr[j].val_arr_len;
			if (panda_sorted_intermediate_keyvals_arr[j].val_arr_len ==0)
				panda_sorted_intermediate_keyvals_arr[j].vals = NULL;
			panda_sorted_intermediate_keyvals_arr[j].val_arr_len += val_arr_len_1;

			val_t *vals = panda_sorted_intermediate_keyvals_arr[j].vals;
			panda_sorted_intermediate_keyvals_arr[j].vals = (val_t*)realloc(vals, sizeof(val_t)*(panda_sorted_intermediate_keyvals_arr[j].val_arr_len));

			for (int k=0;k<val_arr_len_1;k++){
				char *val_0 = (char *)(cpu_sorted_intermediate_keyvals_arr[i].vals[k].val);
				int valSize_0 = cpu_sorted_intermediate_keyvals_arr[i].vals[k].valSize;

				panda_sorted_intermediate_keyvals_arr[j].vals[index+k].val = malloc(sizeof(char)*valSize_0);
				panda_sorted_intermediate_keyvals_arr[j].vals[index+k].valSize = valSize_0;
				memcpy(panda_sorted_intermediate_keyvals_arr[j].vals[index+k].val, val_0, valSize_0);

			}//for
			break;
		}//for

		if (j == d_g_state_0->sorted_keyvals_arr_len){

			if (d_g_state_0->sorted_keyvals_arr_len == 0) panda_sorted_intermediate_keyvals_arr = NULL;

			val_t *vals = cpu_sorted_intermediate_keyvals_arr[i].vals;
			int val_arr_len = cpu_sorted_intermediate_keyvals_arr[i].val_arr_len;

			d_g_state_0->sorted_keyvals_arr_len++;
			
			panda_sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(panda_sorted_intermediate_keyvals_arr, 
				sizeof(keyvals_t)*(d_g_state_0->sorted_keyvals_arr_len));

			int index = d_g_state_0->sorted_keyvals_arr_len-1;
			keyvals_t* kvals_p = (keyvals_t *)&(panda_sorted_intermediate_keyvals_arr[index]);

			kvals_p->keySize = keySize_1;
			kvals_p->key = malloc(sizeof(char)*keySize_1);
			memcpy(kvals_p->key, key_1, keySize_1);

			kvals_p->vals = (val_t *)malloc(sizeof(val_t)*val_arr_len);
			kvals_p->val_arr_len = val_arr_len;

			for (int k=0; k < val_arr_len; k++){
				char *val_0 = (char *)(cpu_sorted_intermediate_keyvals_arr[i].vals[k].val);
				int valSize_0 = cpu_sorted_intermediate_keyvals_arr[i].vals[k].valSize;

				kvals_p->vals[k].valSize = valSize_0;
				kvals_p->vals[k].val = (char *)malloc(sizeof(char)*valSize_0);
				memcpy(kvals_p->vals[k].val,val_0, valSize_0);

			}//for
		}//if (j == sorted_key_arr_len){
	}//if
	d_g_state_0->sorted_intermediate_keyvals_arr = cpu_sorted_intermediate_keyvals_arr;
	ShowLog("CPU_GROUP_ID:[%d] DONE. Sorted len:%d",d_g_state_1->cpu_group_id, d_g_state_0->sorted_keyvals_arr_len);
}
#endif


void PandaExecuteShuffleMergeOnGPU(panda_node_context *pnc, panda_gpu_context *pgc){
	
	char *sorted_keys_shared_buff_0 = (char *)pgc->sorted_key_vals.h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff_0 = (char *)pgc->sorted_key_vals.h_sorted_vals_shared_buff;

	sorted_keyval_pos_t *keyval_pos_arr_0 = pgc->sorted_key_vals.h_sorted_keyval_pos_arr;
	keyvals_t *sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;

	void *key_0, *key_1;
	int keySize_0, keySize_1;
	bool equal;

	int new_count = 0;
	for (int i=0;i< pgc->sorted_key_vals.d_sorted_keyvals_arr_len;i++){
		key_0 = sorted_keys_shared_buff_0 + keyval_pos_arr_0[i].keyPos;
		keySize_0 = keyval_pos_arr_0[i].keySize;
		int j = 0;
		
		for (; j< pnc->sorted_key_vals.sorted_keyvals_arr_len; j++){

			key_1 = sorted_intermediate_keyvals_arr[j].key;
			keySize_1 = sorted_intermediate_keyvals_arr[j].keySize;

			if(panda_cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;

			val_t *vals = sorted_intermediate_keyvals_arr[j].vals;
			//copy values from gpu to cpu context
			int val_arr_len_0 = keyval_pos_arr_0[i].val_arr_len;
			val_pos_t * val_pos_arr = keyval_pos_arr_0[i].val_pos_arr;

			int index = sorted_intermediate_keyvals_arr[j].val_arr_len;
			sorted_intermediate_keyvals_arr[j].val_arr_len += val_arr_len_0;
			sorted_intermediate_keyvals_arr[j].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[j].val_arr_len));

			for (int k=0; k < val_arr_len_0; k++){

				char *val_0 = sorted_vals_shared_buff_0 + val_pos_arr[k].valPos;
				int valSize_0 = val_pos_arr[k].valSize;

				sorted_intermediate_keyvals_arr[j].vals[index+k].val = (char *)malloc(sizeof(char)*valSize_0);
				sorted_intermediate_keyvals_arr[j].vals[index+k].valSize = valSize_0;
				memcpy(sorted_intermediate_keyvals_arr[j].vals[index+k].val, val_0, valSize_0);

			}//for
			break;
		}//for

		if (j == pnc->sorted_key_vals.sorted_keyvals_arr_len){

			if (pnc->sorted_key_vals.sorted_keyvals_arr_len == 0) sorted_intermediate_keyvals_arr = NULL;
			int val_arr_len =keyval_pos_arr_0[i].val_arr_len;
			val_pos_t * val_pos_arr =keyval_pos_arr_0[i].val_pos_arr;
			pnc->sorted_key_vals.sorted_keyvals_arr_len++;

			sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*(pnc->sorted_key_vals.sorted_keyvals_arr_len));
			int index = pnc->sorted_key_vals.sorted_keyvals_arr_len-1;
			keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);

			kvals_p->keySize = keySize_0;
			kvals_p->key = malloc(sizeof(char)*keySize_0);

			memcpy(kvals_p->key, key_0, keySize_0);
			kvals_p->vals = (val_t *)malloc(sizeof(val_t)*val_arr_len);
			kvals_p->val_arr_len = val_arr_len;

			for (int k=0; k < val_arr_len; k++){

				char *val_0 = sorted_vals_shared_buff_0 + val_pos_arr[k].valPos;
				int valSize_0 = val_pos_arr[k].valSize;

				kvals_p->vals[k].valSize = valSize_0;
				kvals_p->vals[k].val = (char *)malloc(sizeof(char)*valSize_0);
				memcpy(kvals_p->vals[k].val, val_0, valSize_0);

			}//for
		}//if
	}//if 
	pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;

	return;

}//void


#endif 
