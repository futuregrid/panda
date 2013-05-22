#include <mpi.h>

#include <panda/Chunk.h>
#include <panda/Sorter.h>
#include <panda/Mapper.h>
#include <panda/Reducer.h>
#include <panda/Combiner.h>
#include <panda/Partitioner.h>
#include <panda/PandaMessage.h>
#include <panda/PartialReducer.h>
#include <panda/PandaCPUConfig.h>
#include <panda/PandaGPUConfig.h>
#include <panda/PandaMapReduceJob.h>
#include <panda/EmitConfiguration.h>

#include <cudacpp/DeviceProperties.h>
#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>

#include "Panda.h"

#include <algorithm>
#include <vector>
#include <cstring>
#include <string>

int gCommRank=0;

namespace panda
{

	//TO remove
  void PandaMapReduceJob::determineMaximumSpaceRequirements()
  {
  }//void
			
  void PandaMapReduceJob::StartPandaLocalMergeGPUOutput()
  {
	  ExecutePandaShuffleMergeGPU(this->pNodeContext, this->pGPUContext);
  }//void

  void PandaMapReduceJob::StartPandaSortGPUResults()
  {
	  ExecutePandaGPUShuffle(this->pGPUContext);
  }//int

 int PandaMapReduceJob::StartPandaGPUReduceTasks()
	{

		//InitGPUDevice(thread_info);
		/*panda_context *panda = (panda_context *)(thread_info->panda);
		gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
		int num_gpu_core_groups = d_g_state->num_gpu_core_groups;
		if ( num_gpu_core_groups <= 0){
			ShowError("num_gpu_core_groups == 0 return");
			return NULL;
		}*///if

		//TODO add input record gpu
		//AddReduceInputRecordGPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), thread_info->start_idx, thread_info->end_idx);

		int gpu_id;
		cudaGetDevice(&gpu_id);
		ShowLog("Start GPU Reduce Tasks.  GPU_ID:%d",gpu_id);
		ExecutePandaGPUReduceTasks(this->pGPUContext);
		return 0;
	}// int PandaMapReduceJob


	int PandaMapReduceJob::StartPandaCPUMapTasks()
	{
		panda_cpu_context *pcc = this->pCPUContext;
		panda_node_context *pnc = this->pNodeContext;

		//-------------------------------------------------------
		//		0, Check status of pcc;
		//-------------------------------------------------------

		if (pcc->input_key_vals.num_input_record<0)			{ ShowLog("Error: no any input keys");			exit(-1);}
		if (pcc->input_key_vals.input_keyval_arr == NULL)	{ ShowLog("Error: input_keyval_arr == NULL");	exit(-1);}
		if (pcc->num_cpus_cores <= 0)						{ ShowError("Error: pcc->num_cpus == 0");		exit(-1);}

		pcc->num_cpus_cores = getCPUCoresNum();
		int num_cpus_cores = pcc->num_cpus_cores;
		
		int totalKeySize = 0;
		int totalValSize = 0;
		for(int i=0; i<pcc->input_key_vals.num_input_record; i++){
			totalKeySize += pcc->input_key_vals.input_keyval_arr[i].keySize;
			totalValSize += pcc->input_key_vals.input_keyval_arr[i].valSize;
		}//for

		//ShowLog("CPU_GROUP_ID:[%d] num_input_record:%d, totalKeySize:%d KB totalValSize:%d KB num_cpus:%d", 
		//	d_g_state->cpu_group_id, job_conf->num_input_record, totalKeySize/1024, totalValSize/1024, d_g_state->num_cpus_cores);

		pcc->panda_cpu_task = (pthread_t *)malloc(sizeof(pthread_t)*(num_cpus_cores));
		pcc->panda_cpu_task_info = (panda_cpu_task_info_t *)malloc(sizeof(panda_cpu_task_info_t)*(num_cpus_cores));
		pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p = (keyval_arr_t **)malloc(sizeof(keyval_arr_t*)*pcc->input_key_vals.num_input_record);
		memset(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p, 0, sizeof(keyval_arr_t)*pcc->input_key_vals.num_input_record);

		for (int i=0; i < num_cpus_cores; i++){
			
			pcc->panda_cpu_task_info[i].pcc = (panda_cpu_context  *)(this->pCPUContext);
			pcc->panda_cpu_task_info[i].pnc = (panda_node_context *)(this->pNodeContext);
			pcc->panda_cpu_task_info[i].num_cpus_cores = num_cpus_cores;
			pcc->panda_cpu_task_info[i].start_row_idx = 0;
			pcc->panda_cpu_task_info[i].end_row_idx = 0;
			
		}//for

		ShowLog("Configure the parameters for pcc");
	
		pcc->intermediate_key_vals.intermediate_keyval_total_count = (int *)malloc(pcc->input_key_vals.num_input_record*sizeof(int));
		memset(pcc->intermediate_key_vals.intermediate_keyval_total_count, 0, pcc->input_key_vals.num_input_record * sizeof(int));
	
		keyval_arr_t *d_keyval_arr_p;
		int *count = NULL;
	
		int num_input_record		= pcc->input_key_vals.num_input_record;
		int num_records_per_thread	= (num_input_record)/(num_threads);
		int num_threads				= pcc->num_cpus_cores;
	
		int start_row_idx = 0;
		int end_row_idx = 0;
	
		for (int tid = 0; tid < num_threads; tid++){
	
			end_row_idx = start_row_idx + num_records_per_thread;
			if (tid < (num_input_record % num_threads))
				end_row_idx++;
			pcc->panda_cpu_task_thread_info[tid].start_row_idx = start_row_idx;
			if (end_row_idx > num_input_record) 
				end_row_idx = num_input_record;
			pcc->panda_cpu_task_info[tid].end_row_idx = end_row_idx;
			if (pthread_create(&(pcc->panda_cpu_task_thread[tid]),NULL,ExecutePandaCPUMapThread,(char *)&(pcc->panda_cpu_task_thread_info[tid])) != 0) 
				ShowError("Thread creation failed Tid:%d!",tid);
			start_row_idx = end_row_idx;

		}//for
	
		for (int tid = 0;tid<num_threads;tid++){
			void *exitstat;
			if (pthread_join(pcc->panda_cpu_task_thread[tid],&exitstat)!=0)
				ShowError("joining failed tid:%d",tid);
		}//for
	
	}//int PandaMapReduceJob::StartPandaCPUMapTasks()


//void InitGPUCardMapReduce(gpu_card_context* d_g_state)
 int PandaMapReduceJob::StartPandaGPUMapTasks()
	{		

	panda_gpu_context *pgc = this->pGPUContext;

	//-------------------------------------------------------
	//0, Check status of pgc;
	//-------------------------------------------------------
			
	if (pgc->input_key_vals.num_input_record<0)			{ ShowLog("Error: no any input keys"); exit(-1);}
	if (pgc->input_key_vals.h_input_keyval_arr == NULL) { ShowLog("Error: h_input_keyval_arr == NULL"); exit(-1);}

	//if (pgc->input_key_vals.num_mappers<=0) {pgc->num_mappers = (NUM_BLOCKS)*(NUM_THREADS);}
	//if (pgc->input_key_vals.num_reducers<=0) {pgc->num_reducers = (NUM_BLOCKS)*(NUM_THREADS);}

	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------

	keyval_arr_t *h_keyval_arr_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*pgc->input_key_vals.num_input_record);
	keyval_arr_t *d_keyval_arr_arr;
	checkCudaErrors(cudaMalloc((void**)&(d_keyval_arr_arr),pgc->input_key_vals.num_input_record*sizeof(keyval_arr_t)));
	
	for (int i=0; i<pgc->input_key_vals.num_input_record;i++){
		h_keyval_arr_arr[i].arr = NULL;
		h_keyval_arr_arr[i].arr_len = 0;
	}//for

	keyval_arr_t **d_keyval_arr_arr_p;
	checkCudaErrors(cudaMalloc((void***)&(d_keyval_arr_arr_p),pgc->input_key_vals.num_input_record*sizeof(keyval_arr_t*)));
	pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p = d_keyval_arr_arr_p;
		
	int *count = NULL;
	checkCudaErrors(cudaMalloc((void**)(&count),pgc->input_key_vals.num_input_record*sizeof(int)));
	pgc->intermediate_key_vals.d_intermediate_keyval_total_count = count;
	checkCudaErrors(cudaMemset(pgc->intermediate_key_vals.d_intermediate_keyval_total_count,0,
		pgc->input_key_vals.num_input_record*sizeof(int)));

	//----------------------------------------------
	//3, determine the number of threads to run
	//----------------------------------------------
	
	//--------------------------------------------------
	//4, start_row_id map
	//Note: DO *NOT* set large number of threads within block (512), which lead to too many invocation of malloc in the kernel. 
	//--------------------------------------------------

	cudaThreadSynchronize();
	
	int numGPUCores = getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
    	dim3 grids(numBlocks, 1);
	int total_gpu_threads = (grids.x*grids.y*blocks.x*blocks.y);
	ShowLog("GridDim.X:%d GridDim.Y:%d BlockDim.X:%d BlockDim.Y:%d TotalGPUThreads:%d",grids.x,grids.y,blocks.x,blocks.y,total_gpu_threads);

	cudaDeviceSynchronize();
	double t1 = PandaTimer();
	
	StartPandaGPUMapPartitioner(*pgc,grids,blocks);
	
	cudaThreadSynchronize();
	double t2 = PandaTimer();

	int num_records_per_thread = (pgc->input_key_vals.num_input_record + (total_gpu_threads)-1)/(total_gpu_threads);
	int totalIter = num_records_per_thread;
	//ShowLog("GPUMapPartitioner:%f totalIter:%d",t2-t1, totalIter);

	for (int iter = 0; iter< totalIter; iter++){

		double t3 = PandaTimer();
		RunGPUMapTasksHost(*pgc, totalIter -1 - iter, totalIter, grids,blocks);
		cudaThreadSynchronize();
		double t4 = PandaTimer();
		size_t total_mem,avail_mem;
		checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
		//ShowLog("GPU_ID:[%d] RunGPUMapTasks take %f sec at iter [%d/%d] remain %d mb GPU mem processed",
		//	pgc->gpu_id, t4-t3,iter,totalIter, avail_mem/1024/1024);

	}//for
	//ShowLog("GPU_ID:[%d] Done %d Tasks",pgc->gpu_id,pgc->num_input_record);
	return 0;
}//int 


void PandaMapReduceJob::InitPandaCPUMapReduce()
{

	this->pCPUContext = CreatePandaCPUContext();
	this->pCPUContext->input_key_vals.num_input_record = mapTasks.size();
	this->pCPUContext->input_key_vals.input_keyval_arr = 	(keyval_t *)malloc(mapTasks.size()*sizeof(keyval_t));

	for (unsigned int i= 0;i<mapTasks.size();i++){

		void *key = this->mapTasks[i]->key;
		int keySize = this->mapTasks[i]->keySize;
		void *val = this->mapTasks[i]->val;
		int valSize = this->mapTasks[i]->valSize;
		this->pCPUContext->input_key_vals.input_keyval_arr[i].key = key;
		this->pCPUContext->input_key_vals.input_keyval_arr[i].keySize = keySize;
		this->pCPUContext->input_key_vals.input_keyval_arr[i].val = val;
		this->pCPUContext->input_key_vals.input_keyval_arr[i].valSize = valSize;

	}//for

	panda_cpu_context* pcc = this->pCPUContext;

	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<pcc->input_key_vals.num_input_record;i++){
		totalKeySize += pcc->input_key_vals.input_keyval_arr[i].keySize;
		totalValSize += pcc->input_key_vals.input_keyval_arr[i].valSize;
	}//for

	//ShowLog("GPU_ID:[%d] copy %d input records from Host to GPU memory totalKeySize:%d KB totalValSize:%d KB",
	//	pgc->gpu_id, pgc->num_input_record, totalKeySize/1024, totalValSize/1024);
	
	void *input_vals_shared_buff = malloc(totalValSize);
	void *input_keys_shared_buff = malloc(totalKeySize);
	
	keyval_pos_t *input_keyval_pos_arr = 
		(keyval_pos_t *)malloc(sizeof(keyval_pos_t)*pcc->input_key_vals.num_input_record);
	
	int keyPos  = 0;
	int valPos  = 0;
	int keySize = 0;
	int valSize = 0;
	
	for(int i=0;i<pcc->input_key_vals.num_input_record;i++){
		
		keySize = pcc->input_key_vals.input_keyval_arr[i].keySize;
		valSize = pcc->input_key_vals.input_keyval_arr[i].valSize;
		
		memcpy((char *)input_keys_shared_buff + keyPos,(char *)(pcc->input_key_vals.input_keyval_arr[i].key), keySize);
		memcpy((char *)input_vals_shared_buff + valPos,(char *)(pcc->input_key_vals.input_keyval_arr[i].val), valSize);
		
		input_keyval_pos_arr[i].keySize = keySize;
		input_keyval_pos_arr[i].keyPos = keyPos;
		input_keyval_pos_arr[i].valPos = valPos;
		input_keyval_pos_arr[i].valSize = valSize;

		keyPos += keySize;	
		valPos += valSize;

	}//for

}//void


void PandaMapReduceJob::InitPandaGPUMapReduce()
{

	this->pGPUContext = CreatePandaGPUContext();
	this->pGPUContext->input_key_vals.num_input_record = mapTasks.size();//Ratio
	this->pGPUContext->input_key_vals.h_input_keyval_arr = 	(keyval_t *)malloc(mapTasks.size()*sizeof(keyval_t));

	for (unsigned int i= 0;i<mapTasks.size();i++){
		void *key = this->mapTasks[i]->key;
		int keySize = this->mapTasks[i]->keySize;
		void *val = this->mapTasks[i]->val;
		int valSize = this->mapTasks[i]->valSize;
		this->pGPUContext->input_key_vals.h_input_keyval_arr[i].key = key;
		this->pGPUContext->input_key_vals.h_input_keyval_arr[i].keySize = keySize;
		this->pGPUContext->input_key_vals.h_input_keyval_arr[i].val = val;			//didn't use memory copy
		this->pGPUContext->input_key_vals.h_input_keyval_arr[i].valSize = valSize;
	}//for

	panda_gpu_context* pgc = this->pGPUContext;

	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<pgc->input_key_vals.num_input_record;i++){
		totalKeySize += pgc->input_key_vals.h_input_keyval_arr[i].keySize;
		totalValSize += pgc->input_key_vals.h_input_keyval_arr[i].valSize;
	}//for

	//ShowLog("GPU_ID:[%d] copy %d input records from Host to GPU memory totalKeySize:%d KB totalValSize:%d KB",
	//	pgc->gpu_id, pgc->num_input_record, totalKeySize/1024, totalValSize/1024);

	void *input_vals_shared_buff = malloc(totalValSize);
	void *input_keys_shared_buff = malloc(totalKeySize);
	keyval_pos_t *input_keyval_pos_arr = 
		(keyval_pos_t *)malloc(sizeof(keyval_pos_t)*pgc->input_key_vals.num_input_record);
	
	int keyPos = 0;
	int valPos = 0;
	int keySize = 0;
	int valSize = 0;
	
	for(int i=0;i<pgc->input_key_vals.num_input_record;i++){
		
		keySize = pgc->input_key_vals.h_input_keyval_arr[i].keySize;
		valSize = pgc->input_key_vals.h_input_keyval_arr[i].valSize;
		
		memcpy((char *)input_keys_shared_buff + keyPos,(char *)(pgc->input_key_vals.h_input_keyval_arr[i].key), keySize);
		memcpy((char *)input_vals_shared_buff + valPos,(char *)(pgc->input_key_vals.h_input_keyval_arr[i].val), valSize);
		
		input_keyval_pos_arr[i].keySize = keySize;
		input_keyval_pos_arr[i].keyPos = keyPos;
		input_keyval_pos_arr[i].valPos = valPos;
		input_keyval_pos_arr[i].valSize = valSize;

		keyPos += keySize;	
		valPos += valSize;

	}//for


	checkCudaErrors(cudaMalloc((void **)&pgc->input_key_vals.d_input_keyval_pos_arr,sizeof(keyval_pos_t)*pgc->input_key_vals.num_input_record));
	checkCudaErrors(cudaMalloc((void **)&pgc->input_key_vals.d_input_keys_shared_buff, totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&pgc->input_key_vals.d_input_vals_shared_buff, totalValSize));
	checkCudaErrors(cudaMemcpy(pgc->input_key_vals.d_input_keyval_pos_arr, input_keyval_pos_arr,sizeof(keyval_pos_t)*pgc->input_key_vals.num_input_record ,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pgc->input_key_vals.d_input_keys_shared_buff, input_keys_shared_buff,totalKeySize ,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pgc->input_key_vals.d_input_vals_shared_buff, input_vals_shared_buff,totalValSize ,cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(pgc->d_input_keyval_arr,h_buff,sizeof(keyval_t)*pgc->num_input_record,cudaMemcpyHostToDevice));
	cudaThreadSynchronize(); 
	//pgc->iterative_support = true;

}//void



	void PandaMapReduceJob::InitPandaRuntime(){
	
		this->pNodeContext = new panda_node_context;
		if (this->pNodeContext == NULL) exit(-1);
		memset(this->pNodeContext, 0, sizeof(panda_node_context));

		if (this->commRank == 0){
			ShowLog("commRank:%d, commSize:%d",this->commRank, this->commSize);
			this->pRuntimeContext = new panda_runtime_context;
		}//if
		else
			this->pRuntimeContext = NULL;

		//StartPandaMessageThread();

	}//void

  //To be removed
  void PandaMapReduceJob::allocateMapVariables()
  {

  }

  //To be removed
  void PandaMapReduceJob::freeMapVariables()
  {
    
  }

  void PandaMapReduceJob::StartPandaMessageThread()
  {
    MessageThread = new oscpp::Thread(messager);
    MessageThread->start();
  }

  void PandaMapReduceJob::mapChunkExecute(const unsigned int chunkIndex,
                                              PandaGPUConfig & config,
                                              void * const memPool)
  {
	  
    chunks[chunkIndex]->stageAsync(memPool, kernelStream);
    mapper->executeOnGPUAsync(chunks[chunkIndex], config, memPool, kernelStream, memcpyStream);
    if (partialReducer != NULL) partialReducer->executeOnGPUAsync(emitConfigs[chunkIndex], config.keySpace, config.valueSpace, memPool, gpuKeyCounts, gpuKeyOffsets, gpuValCounts, gpuValOffsets, kernelStream);
  }

  void PandaMapReduceJob::mapChunkMemcpy(const unsigned int chunkIndex,
                                             const void * const gpuKeySpace,
                                             const void * const gpuValueSpace)
  {
    if (!accumMap || chunkIndex + 1 == chunks.size())
    {
      cudacpp::Runtime::memcpyDtoHAsync(cpuKeys,        gpuKeySpace,   maxKeySpace,             memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuVals,        gpuValueSpace, maxValSpace,             memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuKeyOffsets,  gpuKeyOffsets, commSize * sizeof(int),  memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuValOffsets,  gpuValOffsets, commSize * sizeof(int),  memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuKeyCounts,   gpuKeyCounts,  commSize * sizeof(int),  memcpyStream);
      cudacpp::Runtime::memcpyDtoHAsync(cpuValCounts,   gpuValCounts,  commSize * sizeof(int),  memcpyStream);
    }//if
  }//void

  void PandaMapReduceJob::mapChunkPartition(const unsigned int chunkIndex,
                                                void * const memPool,
                                                PandaGPUConfig & config)
  {
    if (combiner == NULL && accumMap && chunkIndex == chunks.size() - 1)
    {
      memcpyStream->sync();
      partitionSub(memPool,
                   config.keySpace,
                   config.valueSpace,
                   emitConfigs[chunkIndex].getIndexCount(),
                   emitConfigs[chunkIndex].getKeySize(),
                   emitConfigs[chunkIndex].getValueSize());
    }
    if (combiner == NULL && !accumMap) partitionChunk(chunkIndex);
  }//void

  void PandaMapReduceJob::queueChunk(const unsigned int chunkIndex)
  {
    PandaGPUConfig config;
    void * memPool    = reinterpret_cast<char * >(gpuStaticMems) + (maxStaticMem * chunkIndex) % numBuffers;
    config.keySpace   = reinterpret_cast<char * >(gpuKeys) + (maxKeySpace * chunkIndex) % numBuffers;
    config.valueSpace = reinterpret_cast<char * >(gpuVals) + (maxValSpace * chunkIndex) % numBuffers;
    config.emitInfo.grid.numThreads     = emitConfigs[chunkIndex].getThreadCount();
    config.emitInfo.grid.emitsPerThread = emitConfigs[chunkIndex].getEmitsPerThread();

    mapChunkExecute(chunkIndex, config, memPool);
    mapChunkMemcpy(chunkIndex, config.keySpace, config.valueSpace);
    if (reducer != NULL) mapChunkPartition(chunkIndex, memPool, config);
  }//void

  void PandaMapReduceJob::partitionSubDoGPU(void * const memPool,
                                                void * const keySpace,
                                                void * const valueSpace,
                                                const int numKeys,
                                                const int singleKeySize,
                                                const int singleValSize)
  {
    partitioner->executeOnGPUAsync(numKeys,
                                   singleKeySize, singleValSize,
                                   keySpace,      valueSpace,
                                   gpuKeyOffsets, gpuValOffsets,
                                   gpuKeyCounts,  gpuValCounts,
                                   memPool,
                                   kernelStream);
    cudacpp::Runtime::memcpyDtoH(cpuKeyCounts,  gpuKeyCounts,  sizeof(int) * commSize);
    cudacpp::Runtime::memcpyDtoH(cpuValCounts,  gpuValCounts,  sizeof(int) * commSize);
    cudacpp::Runtime::memcpyDtoH(cpuKeyOffsets, gpuKeyOffsets, sizeof(int) * commSize);
    cudacpp::Runtime::memcpyDtoH(cpuValOffsets, gpuValOffsets, sizeof(int) * commSize);
    cudacpp::Runtime::memcpyDtoH(cpuKeys, keySpace,   singleKeySize * numKeys);
    cudacpp::Runtime::memcpyDtoH(cpuVals, valueSpace, singleValSize * numKeys);
  }//void

  void PandaMapReduceJob::partitionSubDoNullPartitioner(const int numKeys)
  {

    cpuKeyOffsets[0] = cpuValOffsets[0] = 0;
    cpuKeyCounts[0]  = cpuValCounts[0] = numKeys;

    for (int index   = 1; index < commSize; ++index)
    {
      cpuKeyOffsets[index] = cpuValOffsets[index] = 0;
      cpuKeyCounts [index] = cpuValCounts[index]  = 0;
    }//for

  }//void

  //send output results of map tasks
  void PandaMapReduceJob::partitionSubSendData(const int singleKeySize, const int singleValSize)
  {
    int totalKeySize = 0;
    for (int index = 0; index < commSize; ++index)
    {
      int i = (index + commRank) % commSize;
      int keyBytes = cpuKeyCounts[i] * singleKeySize;
      int valBytes = cpuValCounts[i] * singleValSize;
      totalKeySize += keyBytes;

      if (keyBytes + valBytes > 0) // it can happen that we send nothing.
      {
        oscpp::AsyncIORequest * ioReq = messager->sendTo(i,
                                                       reinterpret_cast<char * >(cpuKeys) + cpuKeyOffsets[i] * singleKeySize,
                                                       reinterpret_cast<char * >(cpuVals) + cpuValOffsets[i] * singleValSize,
                                                       keyBytes,
                                                       valBytes);
        sendReqs.push_back(ioReq);
      }
    }
  }//void

  //check whether the intermediate task has been send out
  void PandaMapReduceJob::StartPandaPartitionCheckSends(const bool sync)
  {
    std::vector<oscpp::AsyncIORequest * > newReqs;
    for (unsigned int j = 0; j < sendReqs.size(); ++j)
    {
      if (sync) sendReqs[j]->sync();
      if (sendReqs[j]->query()) delete sendReqs[j];
      else                      newReqs.push_back(sendReqs[j]);
    }
    sendReqs = newReqs;
  }

  //The Hash Partition or Shuffle stage at the program.
  //TODO   3/6/2013
  void PandaMapReduceJob::partitionSub(void * const memPool,
                                           void * const keySpace,
                                           void * const valueSpace,
                                           const int numKeys,
                                           const int singleKeySize,
                                           const int singleValSize)
  {
    StartPandaPartitionCheckSends(false);
    if (partitioner != NULL)
    {
      partitionSubDoGPU(memPool, keySpace, valueSpace, numKeys, singleKeySize, singleValSize);
    }//if
    if (partitioner == NULL) // everything goes to rank 0
    {
      partitionSubDoNullPartitioner(numKeys);
    }//if
    partitionSubSendData(singleKeySize, singleValSize);
    if (syncPartSends) StartPandaPartitionCheckSends(true);
  }//void

  void PandaMapReduceJob::partitionChunk(const unsigned int chunkIndex)
  {
    void * memPool    = reinterpret_cast<char * >(gpuStaticMems) + (maxStaticMem * chunkIndex) % numBuffers;
    void * keySpace   = reinterpret_cast<char * >(gpuKeys) + (maxKeySpace * chunkIndex) % numBuffers;
    void * valueSpace = reinterpret_cast<char * >(gpuVals) + (maxValSpace * chunkIndex) % numBuffers;
    partitionSub(memPool,
                 keySpace,
                 valueSpace,
                 emitConfigs[chunkIndex].getIndexCount(),
                 emitConfigs[chunkIndex].getKeySize(),
                 emitConfigs[chunkIndex].getValueSize());
  }//void

  //To be removed
  void PandaMapReduceJob::saveChunk(const unsigned int chunkIndex)
  {
  }//void

  void PandaMapReduceJob::combine()
  {
  }//void

  //To be removed
  void PandaMapReduceJob::globalPartition()
  {
  }//void

  void PandaMapReduceJob::enqueueAllChunks()
  {
    if (reducer == NULL) messager->MsgFinish();
    for (unsigned int i = 0; i < chunks.size(); ++i)
    {
      queueChunk(i);
      if (combiner != NULL && !accumMap) saveChunk(i);
    }//for

    for (unsigned int i = 0; i < chunks.size(); ++i)
    {
      delete chunks[i];
    }//for

    if (combiner != NULL)
    {
      combine();
    }
    if (combiner != NULL)
    {
      globalPartition();
    }
    chunks.clear();
  }

  void PandaMapReduceJob::StartPandaExitMessager()
  {
    if (messager!=NULL) messager->MsgFinish();

    MessageThread->join();
    delete MessageThread;
	ShowLog("MessageThread Join() completed.");

  }//void

  void PandaMapReduceJob::collectVariablesFromMessageAndKill()
  {
    //if (reducer != NULL)
    messager->MsgFinish();
    MessageThread->join();
    ShowLog("MsgThread Join Complete.");
    delete MessageThread;

    if (sorter != NULL || reducer != NULL)
    {
      messager->getFinalDataSize(keySize, valSize);
      if (keySize > 0)
      {
        keys = cudacpp::Runtime::mallocHost(keySize);
        vals = cudacpp::Runtime::mallocHost(valSize);
        messager->getFinalData(keys, vals);
      }//if
    }//if
  }//void

  //To be removed
  void PandaMapReduceJob::getReduceRunParameters()
  {
  }

  //To be removed
  PandaGPUConfig & PandaMapReduceJob::getReduceConfig(const int index)
  {
    return configs[index % configs.size()];
  }

  //To be removed
  void PandaMapReduceJob::allocateReduceVariables()
  {
  }

  //To be removed
  void PandaMapReduceJob::freeReduceVariables()
  {
  }

  //To be removed
  void PandaMapReduceJob::copyReduceInput(const int index, const int keysSoFar)
  {
  }

  //reserve
  void PandaMapReduceJob::executeReduce(const int index)
  {
    reducer->executeOnGPUAsync(keyCount[index],
                               gpuInputKeys,
                               gpuInputVals,
                               NULL,
                               gpuInputValOffsets,
                               gpuInputValCounts,
                               getReduceConfig(index),
                               kernelStream);
  }

  void PandaMapReduceJob::copyReduceOutput(const int index, const int keysSoFar)
  {
    cudacpp::Runtime::memcpyDtoHAsync(reinterpret_cast<int * >(cpuKeys) + keysSoFar, getReduceConfig(index).keySpace,   keyCount[index] * sizeof(int), memcpyStream);
    cudacpp::Runtime::memcpyDtoHAsync(reinterpret_cast<int * >(cpuVals) + keysSoFar, getReduceConfig(index).valueSpace, keyCount[index] * sizeof(int), memcpyStream);
  }

  //To be removed
  void PandaMapReduceJob::enqueueReductions()
  {
  }

  void PandaMapReduceJob::map()
  {
    StartPandaMessageThread();
    MPI_Barrier(MPI_COMM_WORLD);
    enqueueAllChunks();
    StartPandaPartitionCheckSends(true);
    collectVariablesFromMessageAndKill();
    freeMapVariables();
  }

  void PandaMapReduceJob::sort()
  {
    numUniqueKeys = -1;
    if (sorter->canExecuteOnGPU())  sorter->executeOnGPUAsync(keys, vals, keySize / sizeof(int), numUniqueKeys, &keyOffsets, &valOffsets, &numVals);
    else                            sorter->executeOnCPUAsync(keys, vals, valSize / sizeof(int), numUniqueKeys, &keyOffsets, &valOffsets, &numVals);
  }

  void PandaMapReduceJob::reduce()
  {

    kernelStream = memcpyStream = cudacpp::Stream::nullStream;
    getReduceRunParameters();
    allocateReduceVariables();
    enqueueReductions();
    cudacpp::Runtime::sync();
    freeReduceVariables();
  }

  //To be removed
  void PandaMapReduceJob::collectTimings()
  {
  }

  PandaMapReduceJob::PandaMapReduceJob(int & argc,
                                               char **& argv,
                                               const bool accumulateMapResults,
                                               const bool accumulateReduceResults,
                                               const bool syncOnPartitionSends)
    : MapReduceJob(argc, argv)
  {
    accumMap = accumulateMapResults;
    accumReduce = accumulateReduceResults;
    syncPartSends = syncOnPartitionSends;
  }

  PandaMapReduceJob::~PandaMapReduceJob()
  {
  }//PandaMapReduceJob

  //Significant change is required in this place
  void PandaMapReduceJob::addInput(Chunk * chunk)
  {
    chunks.push_back(chunk);
	addMapTasks(chunk);
  }//void

  //To be removed
  void PandaMapReduceJob::addMapTasks(panda::Chunk *chunk)
  {
	  //MapTask *pMapTask = new MapTask(keySize,key,valSize,val);
	  //mapTasks.push_back(pMapTask);
  }//void


  void PandaMapReduceJob::StartPandaGPUCombiner()
  {
	  ExecutePandaGPUCombiner(this->pGPUContext);
  }//void 
			
  void PandaMapReduceJob::StartPandaSortBucket()
  {
	  ExecutePandaSortBucket(this->pNodeContext);
  }//for

  void PandaMapReduceJob::StartPandaCopyRecvedBucketToGPU()
  {
 	  int end_row_id = this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len;
	  ShowLog("sorted_keyvals_arr_len:%d",end_row_id);
	  AddReduceTask4GPU(this->pGPUContext,this->pNodeContext, 0, end_row_id);
  }//void

  void PandaMapReduceJob::StartPandaAddReduceTask4GPU(int start_row_id, int end_row_id)
  {
	  AddReduceTask4GPU(this->pGPUContext,this->pNodeContext,start_row_id,end_row_id);
  }//void


  int PandaMapReduceJob::GetHash(const char* Key, int KeySize, int commSize)
  {  
        /////FROM : http://courses.cs.vt.edu/~cs2604/spring02/Projects/4/elfhash.cpp
        unsigned long h = 0;
        while(KeySize-- > 0)
        {
                h = (h << 4) + *Key++;
                unsigned long g = h & 0xF0000000L;
                if (g) h ^= g >> 24;
                h &= ~g;
        }//while            
        return (int) ((int)h % commSize);
  }//int

  void PandaMapReduceJob::PandaAddKeyValue2Bucket(int bucketId, const char*key, int keySize, const char*val, int valSize)
  {

	  char * keyBuff = (char *)(this->pNodeContext->buckets.savedKeysBuff.at(bucketId));
	  char * valBuff = (char *)(this->pNodeContext->buckets.savedValsBuff.at(bucketId));

	  int keyBuffSize = this->pNodeContext->buckets.keyBuffSize[bucketId];
	  int valBuffSize = this->pNodeContext->buckets.valBuffSize[bucketId];
	  int *counts = this->pNodeContext->buckets.counts[bucketId];
	  
	  int curlen	 = counts[0];
	  int maxlen	 = counts[1];
	  //printf("bucketId:%d  0:%d 1:%d,2:%d,3:%d  curlen:%d  maxlen:%d\n",bucketId,counts[0],counts[1],counts[2],counts[3],curlen,maxlen);
	  //printf("keyPos.size:%d\n",this->pNodeContext->buckets.keyPos.size());

	  int keyBufflen = counts[2];
	  int valBufflen = counts[3];

	  int *keyPosArray = this->pNodeContext->buckets.keyPos[bucketId];
	  int *valPosArray = this->pNodeContext->buckets.valPos[bucketId];
	  int *keySizeArray = this->pNodeContext->buckets.keySize[bucketId];
	  int *valSizeArray = this->pNodeContext->buckets.valSize[bucketId];

	  if (keyBufflen + keySize >= keyBuffSize){
			char *newKeyBuff = (char*)malloc(2*keyBuffSize);
			//char *)(realloc(keyBuff, 2*keyBuffSize*sizeof(char)));
			memcpy(newKeyBuff, keyBuff, keyBufflen);
			memcpy(newKeyBuff+keyBufflen, key, keySize);
			counts[2] = keyBufflen + keySize;
			this->pNodeContext->buckets.savedKeysBuff[bucketId] = newKeyBuff;
			this->pNodeContext->buckets.keyBuffSize[bucketId]   = 2*keyBuffSize;
			//TODO remove keyBuff in std::vector
			//delete [] keyBuff
	  }else{
			memcpy(keyBuff + keyBufflen, key, keySize);
			counts[2] = keyBufflen+keySize;
	  }//else
	  
	  if (valBufflen + valSize >= valBuffSize){
		    char *newValBuff = (char*)malloc(2*valBuffSize);
			memcpy(newValBuff, valBuff, valBufflen);
			memcpy(newValBuff + valBufflen, val, valSize);
			counts[3] = valBufflen+valSize;
			this->pNodeContext->buckets.savedValsBuff[bucketId] = newValBuff;
			this->pNodeContext->buckets.valBuffSize[bucketId]	= 2*valBuffSize;
			//TODO remove valBuff in std::vector
			//delete [] valBuff;
	  }else{
			memcpy(valBuff + valBufflen, val, valSize);	//
			counts[3] = valBufflen+valSize;				//
	  }//else

	  keyPosArray[curlen]  = keyBufflen;
      valPosArray[curlen]  = valBufflen;
	  keySizeArray[curlen] = keySize;
	  valSizeArray[curlen] = valSize;

	  (counts[0])++;//increase one keyVal pair
	  if(counts[0] >= counts[1]){
		 
		 counts[1] *= 2;
		 int * newKeyPosArray = (int *)malloc(sizeof(int)*counts[1]);
		 int * newValPosArray = (int *)malloc(sizeof(int)*counts[1]);
		 int * newKeySizeArray = (int *)malloc(sizeof(int)*counts[1]);
		 int * newValSizeArray = (int *)malloc(sizeof(int)*counts[1]);

		 memcpy(newKeyPosArray, keyPosArray, sizeof(int)*counts[0]);
		 memcpy(newValPosArray, valPosArray, sizeof(int)*counts[0]);
		 memcpy(newKeySizeArray, keySizeArray, sizeof(int)*counts[0]);
		 memcpy(newValSizeArray, valSizeArray, sizeof(int)*counts[0]);

		 this->pNodeContext->buckets.keyPos[bucketId]  = newKeyPosArray;
		 this->pNodeContext->buckets.valPos[bucketId]  = newValPosArray;
		 this->pNodeContext->buckets.keySize[bucketId] = newKeySizeArray;
		 this->pNodeContext->buckets.valSize[bucketId] = newValSizeArray;

	  }//if
  }//void

  void PandaMapReduceJob::PandaPartitionCheckSends(const bool sync)
  {
    std::vector<oscpp::AsyncIORequest * > newReqs;
    for (unsigned int j = 0; j < sendReqs.size(); ++j)
    {
      if (sync) sendReqs[j]->sync();
      if (sendReqs[j]->query()) delete sendReqs[j];
      else    newReqs.push_back(sendReqs[j]);
    }//for
    sendReqs = newReqs;
  }//void

  void PandaMapReduceJob::StartPandaDoPartitionOnCPU(){

	  int keyBuffSize = 1024;
	  int valBuffSize = 1024;
	  int maxlen	  = 20;

	  this->pNodeContext->buckets.numBuckets  = this->commSize;
	  this->pNodeContext->buckets.keyBuffSize = new int[this->commSize];
	  this->pNodeContext->buckets.valBuffSize = new int[this->commSize];

	  for (int i=0; i<this->commSize; i++){
		  
		  this->pNodeContext->buckets.keyBuffSize[i] = keyBuffSize;
		  this->pNodeContext->buckets.valBuffSize[i] = valBuffSize;

		  char *keyBuff = new char[keyBuffSize];
		  char *valBuff = new char[valBuffSize];
		  this->pNodeContext->buckets.savedKeysBuff.push_back((char*)keyBuff);
		  this->pNodeContext->buckets.savedValsBuff.push_back((char*)valBuff);
		  int *keyPos  = new int[maxlen];
		  int *valPos  = new int[maxlen];
		  int *keySize = new int[maxlen];
		  int *valSize = new int[maxlen];
		  this->pNodeContext->buckets.keyPos.push_back(keyPos);
		  this->pNodeContext->buckets.valPos.push_back(valPos);
		  this->pNodeContext->buckets.valSize.push_back(valSize);
		  this->pNodeContext->buckets.keySize.push_back(keySize);
		  int* counts_i = new int[4];
		  counts_i[0] = 0;		//curlen
		  counts_i[1] = maxlen;	
		  counts_i[2] = 0;		//keybufflen
		  counts_i[3] = 0;		//valbufflen
		  this->pNodeContext->buckets.counts.push_back(counts_i);
	  }//for

	  keyvals_t *sorted_intermediate_keyvals_arr1 = this->pNodeContext->sorted_key_vals.sorted_intermediate_keyvals_arr;
	  for (int i=0; i<this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len; i++){
		char *key	 = (char *)(sorted_intermediate_keyvals_arr1[i].key);
		int keySize  = sorted_intermediate_keyvals_arr1[i].keySize;
		int bucketId = GetHash(key,keySize,this->commSize);
		val_t *vals  = sorted_intermediate_keyvals_arr1[i].vals;
		int len = sorted_intermediate_keyvals_arr1[i].val_arr_len;
		for (int j=0;j<len;j++){
			PandaAddKeyValue2Bucket(bucketId, (char *)key, keySize,(char *)(vals[j].val),vals[j].valSize);
		}//for
	  }//for
	  //ShowLog("PandaAddKeyValue2Bucket Done\n");

  }//void

  void PandaMapReduceJob::StartPandaPartitionSubSendData()
  {

    for (int index = 0; index < commSize; ++index)
    {
	  int curlen	= this->pNodeContext->buckets.counts[index][0];
	  int maxlen	= this->pNodeContext->buckets.counts[index][1];
	  int keySize	= this->pNodeContext->buckets.counts[index][2];
	  int valSize	= this->pNodeContext->buckets.counts[index][3];

	  ShowLog("index:%d curlen:%d maxlen:%d keySize:%d valSize:%d",index,curlen,maxlen,keySize,valSize);

	  char *keyBuff = this->pNodeContext->buckets.savedKeysBuff[index];
	  char *valBuff = this->pNodeContext->buckets.savedValsBuff[index];

	  int *keySizeArray = this->pNodeContext->buckets.keySize[index];
	  int *valSizeArray = this->pNodeContext->buckets.valSize[index];
	  int *keyPosArray  = this->pNodeContext->buckets.keyPos[index];
	  int *valPosArray  = this->pNodeContext->buckets.valPos[index];
	  
	  int *keyPosKeySizeValPosValSize = new int[curlen*4];
	  
	  if(keyPosArray==NULL)  ShowLog("Error");
	  if(valSizeArray==NULL) ShowLog("Error");
	  if(keyPosArray==NULL)  ShowLog("Error");
	  if(valPosArray==NULL)  ShowLog("Error");

	  //store keyPos keySize valPos valSize in a int array.  
	  
	  for (int i=0; i<curlen; i++){
		keyPosKeySizeValPosValSize[i] = keyPosArray[i];
	  }//for
	  for (int i=curlen; i<curlen*2; i++){
		keyPosKeySizeValPosValSize[i] = keySizeArray[i-curlen];
	  }//for
	  for (int i=curlen*2; i<curlen*3; i++){
		keyPosKeySizeValPosValSize[i] = valPosArray[i-2*curlen];
	  }//for
	  for (int i=curlen*3; i<curlen*4; i++){
		keyPosKeySizeValPosValSize[i] = valSizeArray[i-3*curlen];
	  }//for

      int i = (index + commRank + commSize - 1) % commSize;
      
      if (keySize+valSize > 0) // it can happen that we send nothing.
      {
        oscpp::AsyncIORequest * ioReq = messager->sendTo(i,
                                                       	keyBuff,
                                                       	valBuff,
						   								keyPosKeySizeValPosValSize,
                                                       	keySize,
                                                     	valSize,
						   								curlen);
        sendReqs.push_back(ioReq);
      }//if
    }
  }//void

  void PandaMapReduceJob::StartPandaGlobalPartition()
  {

	  PandaPartitionCheckSends(false);
	  StartPandaDoPartitionOnCPU();
	  StartPandaPartitionSubSendData();

      if (syncPartSends) PandaPartitionCheckSends(true);

	  /*this->pNodeContext->buckets.savedKeysBuff.clear();
	  this->pNodeContext->buckets.savedValsBuff.clear();
	  this->pNodeContext->buckets.keyPos.clear();
	  this->pNodeContext->buckets.valPos.clear();
	  this->pNodeContext->buckets.counts.clear();*/

  }//void

  void PandaMapReduceJob::execute()
  {		
		
    if (mapper != NULL)
    {
		
      if (messager        != NULL) messager->MsgInit();
      if (combiner        != NULL) combiner->init();
		
	  StartPandaMessageThread();
	  MPI_Barrier(MPI_COMM_WORLD);
		
	  InitPandaRuntime();
	  if (messager    != NULL) messager->setPnc(this->pNodeContext);
	  
	  InitPandaGPUMapReduce();
	  InitPandaCPUMapReduce();

	  StartPandaGPUMapTasks();
	  StartPandaCPUMapTasks();
	  
	  StartPandaGPUCombiner();
	  StartPandaCPUCombiner();

      //mapper->finalize();
      if (messager        != NULL) messager->MsgFinalize();
      if (partitioner     != NULL) partitioner->finalize();
      if (partialReducer  != NULL) partialReducer->finalize();
      if (combiner        != NULL) combiner->init();

    }//if

	double t3 = PandaTimer();

    if (sorter != NULL)
    {
      //Sorter->init();
      //Sort();
      //Sorter->finalize();
	  //Panda Code
	  StartPandaSortGPUResults();
    }//if

	
	/////////////////////////////////
	//	Shuffle Stage Start
	/////////////////////////////////

	StartPandaLocalMergeGPUOutput();
	MPI_Barrier(MPI_COMM_WORLD);
	StartPandaGlobalPartition();
	StartPandaPartitionCheckSends(true);
	StartPandaExitMessager();

	/////////////////////////////////
	//	Shuffle Stage Done
	/////////////////////////////////

	StartPandaSortBucket();
	StartPandaCopyRecvedBucketToGPU();
	
	if (reducer!=NULL){

		int start_row_id = 0;
		int end_row_id = this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len;

		StartPandaAddReduceTask4GPU(start_row_id, end_row_id);
		StartPandaGPUReduceTasks();

	}//if

	/* if (reducer != NULL && numUniqueKeys > 0)
    {
      reducer->init();
      reduce();
      totalTimer.stop();
      reducer->finalize();
    } */
    MPI_Barrier(MPI_COMM_WORLD);

  }
}
