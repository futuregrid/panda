#include <mpi.h>
#include <panda/PandaMapReduceJob.h>
#include <panda/Combiner.h>
#include <panda/Chunk.h>
#include <panda/EmitConfiguration.h>
#include <panda/PandaCPUConfig.h>
#include <panda/PandaGPUConfig.h>
#include <panda/Mapper.h>
#include <panda/PartialReducer.h>
#include <panda/Partitioner.h>
#include <panda/Reducer.h>
#include <panda/Sorter.h>
#include <cudacpp/DeviceProperties.h>
#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>

#include "Panda.h"

#include <algorithm>
#include <vector>
#include <cstring>
#include <string>

namespace panda
{
	
  void PandaMapReduceJob::determineMaximumSpaceRequirements()
  {
    const int combinerMem = combiner == NULL ? 0 : combiner->getMemoryRequirementsOnGPU();
    maxStaticMem = std::max(maxStaticMem, combinerMem);
    for (unsigned int i = 0; i < chunks.size(); ++i)
    {
      EmitConfiguration emitConfig = mapper->getEmitConfiguration(chunks[i]);
      emitConfigs.push_back(emitConfig);
      const int chunkMem    = chunks[i]->getMemoryRequirementsOnGPU();
      const int partMem     = partitioner     == NULL ? 0 : partitioner->getMemoryRequirementsOnGPU(emitConfig);
      const int partialMem  = partialReducer  == NULL ? 0 : partialReducer->getMemoryRequirementsOnGPU(emitConfig);
      maxStaticMem  = std::max(maxStaticMem,  chunkMem + std::max(partMem, partialMem));
      maxKeySpace   = std::max(maxKeySpace,  emitConfig.getKeySpace());
      maxValSpace   = std::max(maxValSpace,  emitConfig.getValueSpace());
    }
    maxStaticMem = std::max(maxStaticMem, 1);
    maxKeySpace  = std::max(maxKeySpace,  1);
    maxValSpace  = std::max(maxValSpace,  1);
			
  }//void
			
 int PandaMapReduceJob::StartPandaGPUMapTasks(panda_gpu_context *pgc)
	{		
			
	//-------------------------------------------------------
	//0, Check status of pgc;
	//-------------------------------------------------------
			
	//ShowLog("GPU_ID:[%d]  num_input_record %d", pgc->gpu_id, pgc->num_input_record);
	if (pgc->input_key_vals.num_input_record<0) { ShowLog("Error: no any input keys"); exit(-1);}
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
	checkCudaErrors(cudaMalloc((void**)&(count),pgc->input_key_vals.num_input_record*sizeof(int)));
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
	//ShowLog("GridDim.X:%d GridDim.Y:%d BlockDim.X:%d BlockDim.Y:%d TotalGPUThreads:%d",grids.x,grids.y,blocks.x,blocks.y,total_gpu_threads);

	cudaDeviceSynchronize();
	double t1 = PandaTimer();
	
	//GPUMapPartitionerHost<<<grids,blocks>>>(*pgc);
	GPUMapPartitionerHost(*pgc,grids,blocks);

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

  void PandaMapReduceJob::InitPandaGPUMapReduce(panda_gpu_context* pgc)
{	
	
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


  void PandaMapReduceJob::InitMapTasks(){
	
	//ShouldMove to InitPanda
	this->pNodeContext = new panda_node_context;
	if (this->pNodeContext == NULL) exit(-1);
	memset(this->pNodeContext, 0, sizeof(panda_node_context));

	this->pGPUContext = CreatePandaGPUContext();
	this->pGPUContext->input_key_vals.num_input_record = mapTasks.size();
	this->pGPUContext->input_key_vals.h_input_keyval_arr = 	(keyval_t *)malloc(mapTasks.size()*sizeof(keyval_t));
	//(keyval_t *)realloc(pgc->h_input_keyval_arr, sizeof(keyval_t)*(len + end_row_id - start_row_id));
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

  }

  void PandaMapReduceJob::allocateMapVariables()
  {
    gpuKeys       = cudacpp::Runtime::mallocDevice    (numBuffers * maxKeySpace);
    gpuVals       = cudacpp::Runtime::mallocDevice	  (numBuffers * maxValSpace);
    gpuStaticMems = cudacpp::Runtime::mallocDevice    (numBuffers * maxStaticMem);
    cpuKeys       = cudacpp::Runtime::mallocHost(numBuffers * maxKeySpace);
    cpuVals       = cudacpp::Runtime::mallocHost(numBuffers * maxValSpace);
	//commSize
    gpuKeyOffsets = (int * )(cudacpp::Runtime::mallocDevice    (numBuffers * commSize * sizeof(int) * 4));
	//gpuKeyOffsets = (int *)malloc(numBuffers * commSize * sizeof(int) * 4);
    gpuValOffsets = gpuKeyOffsets + commSize * 1;
    gpuKeyCounts  = gpuKeyOffsets + commSize * 2;
    gpuValCounts  = gpuKeyOffsets + commSize * 3;
    cpuKeyOffsets = (int * )(cudacpp::Runtime::mallocHost(numBuffers * commSize * sizeof(int) * 4));
	
	//cpuKeyOffsets = (int *)malloc(numBuffers * commSize * sizeof(int) * 4);
	//cpuKeyOffsets[0] = 0;
    cpuValOffsets = cpuKeyOffsets + commSize * 1;
    cpuKeyCounts  = cpuKeyOffsets + commSize * 2;
    cpuValCounts  = cpuKeyOffsets + commSize * 3;

	
  }

  void PandaMapReduceJob::freeMapVariables()
  {
    cudacpp::Runtime::free    (gpuKeys);
    cudacpp::Runtime::free    (gpuVals);
    cudacpp::Runtime::free    (gpuStaticMems);
    cudacpp::Runtime::freeHost(cpuKeys);
    cudacpp::Runtime::freeHost(cpuVals);
    cudacpp::Runtime::free    (gpuKeyOffsets);
    cudacpp::Runtime::freeHost(cpuKeyOffsets);
  }
  void PandaMapReduceJob::startMessageThread()
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
  }
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
  }
  void PandaMapReduceJob::partitionSubDoNullPartitioner(const int numKeys)
  {
    cpuKeyOffsets[0] = cpuValOffsets[0] = 0;
    cpuKeyCounts[0] = cpuValCounts[0] = numKeys;
    for (int index = 1; index < commSize; ++index)
    {
      cpuKeyOffsets[index] = cpuValOffsets[index] = 0;
      cpuKeyCounts [index] = cpuValCounts [index] = 0;
    }
  }

  //send  output results of map tasks
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
  }
  //check whether the intermediate task has been send out
  void PandaMapReduceJob::partitionCheckSends(const bool sync)
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
    partitionCheckSends(false);
    if (partitioner != NULL)
    {
      partitionSubDoGPU(memPool, keySpace, valueSpace, numKeys, singleKeySize, singleValSize);
    }
    if (partitioner == NULL) // everything goes to rank 0
    {
      partitionSubDoNullPartitioner(numKeys);
    }
    partitionSubSendData(singleKeySize, singleValSize);
    if (syncPartSends) partitionCheckSends(true);
  }
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
  }
  void PandaMapReduceJob::saveChunk(const unsigned int chunkIndex)
  {
    const int numKeys = emitConfigs[chunkIndex].getIndexCount();
    void * keySpace = reinterpret_cast<char * >(cpuKeys) + (maxKeySpace * chunkIndex) % numBuffers;
    void * valSpace = reinterpret_cast<char * >(cpuVals) + (maxValSpace * chunkIndex) % numBuffers;
    void * keysToSave = new char[numKeys * emitConfigs[chunkIndex].getKeySize()];
    void * valsToSave = new char[numKeys * emitConfigs[chunkIndex].getValueSize()];
    memcpy(keysToSave, keySpace, numKeys * emitConfigs[chunkIndex].getKeySize());
    memcpy(valsToSave, valSpace, numKeys * emitConfigs[chunkIndex].getValueSize());
    savedKeys.push_back(keySpace);
    savedVals.push_back(valSpace);
    keyAndValCount.push_back(numKeys);
  }
  void PandaMapReduceJob::combine()
  {
  }
  void PandaMapReduceJob::globalPartition()
  {
    const int singleKeySize = emitConfigs[0].getKeySize();
    const int singleValSize = emitConfigs[0].getValueSize();
    int numKeys = 0, totalKeySize, totalValSize;
    for (unsigned int i = 0; i < keyAndValCount.size(); ++i)
    {
      numKeys += keyAndValCount[i];
    }
    totalKeySize = numKeys * singleKeySize;
    totalValSize = numKeys * singleValSize;
    char * gpuFullKeys = reinterpret_cast<char * >(cudacpp::Runtime::mallocDevice(totalKeySize));
    char * gpuFullVals = reinterpret_cast<char * >(cudacpp::Runtime::mallocDevice(totalValSize));

    int offset = 0;
    for (unsigned int i = 0; i < keyAndValCount.size(); ++i)
    {
      cudacpp::Runtime::memcpyHtoD(gpuFullKeys + offset, savedKeys[i], keyAndValCount[i] * singleKeySize);
      cudacpp::Runtime::memcpyHtoD(gpuFullVals + offset, savedVals[i], keyAndValCount[i] * singleValSize);
    }
    partitionSub(gpuStaticMems, gpuFullKeys, gpuFullVals, numKeys, singleKeySize, singleValSize);
    cudacpp::Runtime::free(gpuFullKeys);
    cudacpp::Runtime::free(gpuFullVals);
  }
  void PandaMapReduceJob::enqueueAllChunks()
  {
    if (reducer == NULL) messager->finish();
    for (unsigned int i = 0; i < chunks.size(); ++i)
    {
      queueChunk(i);
      if (combiner != NULL && !accumMap) saveChunk(i);
    }
    for (unsigned int i = 0; i < chunks.size(); ++i)
    {
      delete chunks[i];
    }
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

  void PandaMapReduceJob::collectVariablesFromMessageAndKill()
  {
    if (reducer != NULL) messager->finish();
    MessageThread->join();
    delete MessageThread;
    if (sorter != NULL || reducer != NULL)
    {
      messager->getFinalDataSize(keySize, valSize);
      if (keySize > 0)
      {
        keys = cudacpp::Runtime::mallocHost(keySize);
        vals = cudacpp::Runtime::mallocHost(valSize);
        messager->getFinalData(keys, vals);
      }
    }
  }
  void PandaMapReduceJob::getReduceRunParameters()
  {
    maxInputKeySpace = maxInputValSpace = maxInputValOffsetSpace = maxInputNumValsSpace = 0;

    int keysSoFar = 0;
    while (keysSoFar < numUniqueKeys)
    {
      char * tKeys = reinterpret_cast<char * >(keys) + keysSoFar * keySize;
      int numKeysToProcess;
      EmitConfiguration emitConfig = reducer->getEmitConfiguration(tKeys, numVals + keysSoFar, numUniqueKeys - keysSoFar, numKeysToProcess);
      maxKeySpace             = std::max(maxKeySpace,             emitConfig.getKeySpace());
      maxValSpace             = std::max(maxValSpace,             emitConfig.getValueSpace());
      maxInputKeySpace        = std::max(maxInputKeySpace,        numKeysToProcess * static_cast<int>(sizeof(int)));
      maxInputValSpace        = std::max(maxInputValSpace,        (valOffsets[keysSoFar + numKeysToProcess - 1] - valOffsets[keysSoFar] + numVals[keysSoFar + numKeysToProcess - 1]) * static_cast<int>(sizeof(int)));
      maxInputValOffsetSpace  = std::max(maxInputValOffsetSpace,  numKeysToProcess * static_cast<int>(sizeof(int)));
      maxInputNumValsSpace    = std::max(maxInputNumValsSpace,    numKeysToProcess * static_cast<int>(sizeof(int)));
      keysSoFar += numKeysToProcess;
      emitConfigs.push_back(emitConfig);
      keyCount.push_back(numKeysToProcess);
    }
  }
  PandaGPUConfig & PandaMapReduceJob::getReduceConfig(const int index)
  {
    return configs[index % configs.size()];
  }
  void PandaMapReduceJob::allocateReduceVariables()
  {
    gpuKeys             = cudacpp::Runtime::mallocDevice    (numBuffers * maxKeySpace);
    gpuVals             = cudacpp::Runtime::mallocDevice    (numBuffers * maxValSpace);
    gpuInputKeys        = cudacpp::Runtime::mallocDevice    (numBuffers * maxInputKeySpace);
    gpuInputVals        = cudacpp::Runtime::mallocDevice    (numBuffers * maxInputValSpace);
    gpuInputValOffsets  = reinterpret_cast<int * >(cudacpp::Runtime::mallocDevice(numBuffers * maxInputValOffsetSpace));
    gpuInputValCounts   = reinterpret_cast<int * >(cudacpp::Runtime::mallocDevice(numBuffers * maxInputNumValsSpace));
    cpuKeys             = cudacpp::Runtime::mallocHost(numBuffers * maxKeySpace);
    cpuVals             = cudacpp::Runtime::mallocHost(numBuffers * maxValSpace);
  }
  void PandaMapReduceJob::freeReduceVariables()
  {
    if (keys               != NULL) cudacpp::Runtime::freeHost(keys);
    if (vals               != NULL) cudacpp::Runtime::freeHost(vals);
    if (gpuKeys            != NULL) cudacpp::Runtime::free    (gpuKeys);
    if (gpuVals            != NULL) cudacpp::Runtime::free    (gpuVals);
    if (gpuInputKeys       != NULL) cudacpp::Runtime::free    (gpuInputKeys);
    if (gpuInputVals       != NULL) cudacpp::Runtime::free    (gpuInputVals);
    if (gpuInputValOffsets != NULL) cudacpp::Runtime::free    (gpuInputValOffsets);
    if (gpuInputValCounts  != NULL) cudacpp::Runtime::free    (gpuInputValCounts);
    if (cpuKeys            != NULL) cudacpp::Runtime::freeHost(cpuKeys);
    if (cpuVals            != NULL) cudacpp::Runtime::freeHost(cpuVals);
    if (valOffsets         != NULL) cudacpp::Runtime::freeHost(valOffsets);
    if (numVals            != NULL) cudacpp::Runtime::freeHost(numVals);
  }
  void PandaMapReduceJob::copyReduceInput(const int index, const int keysSoFar)
  {
    const int keyCount = std::min(numUniqueKeys, emitConfigs[index].getIndexCount());
    const int keyBytesToCopy  = keyCount * sizeof(int);
    const int valBytesToCopy  = (valOffsets[keysSoFar + keyCount - 1] - valOffsets[keysSoFar] + numVals[keysSoFar + keyCount - 1]) * sizeof(int);

    int numValsToProcess = 0;
    for (int i = 0; i < keyCount; ++i)
    {
      numValsToProcess += numVals[keysSoFar + i];
    }

    cudacpp::Runtime::memcpyHtoDAsync(gpuInputKeys, reinterpret_cast<int * >(keys) + keysSoFar,             keyBytesToCopy, memcpyStream);
    cudacpp::Runtime::memcpyHtoDAsync(gpuInputVals, reinterpret_cast<int * >(vals) + valOffsets[keysSoFar], valBytesToCopy, memcpyStream);
    cudacpp::Runtime::memcpyHtoDAsync(gpuInputValOffsets, valOffsets + keysSoFar, keyCount * sizeof(int), memcpyStream);
    cudacpp::Runtime::memcpyHtoDAsync(gpuInputValCounts,  numVals    + keysSoFar, keyCount * sizeof(int), memcpyStream);
  }
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
  void PandaMapReduceJob::enqueueReductions()
  {
    int keysSoFar = 0;
    for (unsigned int i = 0; i < emitConfigs.size(); ++i)
    {
      EmitConfiguration & emitConfig        = emitConfigs[i];
      getReduceConfig(i).emitInfo.grid.numThreads       = emitConfig.getThreadCount();
      getReduceConfig(i).emitInfo.grid.emitsPerThread   = emitConfig.getEmitsPerThread();

      copyReduceInput(i, keysSoFar);
      cudacpp::Runtime::sync();
      executeReduce(i);
      cudacpp::Runtime::sync();
      copyReduceOutput(i, keysSoFar);
      cudacpp::Runtime::sync();

      cudacpp::Stream::nullStream->sync();

      keysSoFar += keyCount[i];
    }
  }
  void PandaMapReduceJob::map()
  {
    emitConfigs.clear();
    savedKeys.clear();
    savedVals.clear();
    keyAndValCount.clear();
    cudacpp::DeviceProperties * props = cudacpp::DeviceProperties::get(getDeviceNumber());
    kernelStream = memcpyStream = cudacpp::Stream::nullStream;

    maxStaticMem  = 0;
    maxKeySpace = 0;
    maxValSpace = 0;

    // int maxCountSpace = commSize * sizeof(int) * 2;
    // int maxMem        = props->getTotalMemory() - 10 * 1048576;
    // const int maxReq  = maxStaticMem + maxKeySpace + maxValSpace + maxCountSpace + commSize * sizeof(int) * 4;
    numBuffers        = 1; // (accumMap ? 1 : std::min(static_cast<int>(chunks.size()), maxMem / maxReq));

    delete props;
    props = NULL;

    determineMaximumSpaceRequirements();

    keySize = valSize = 0;
    keySpace = maxKeySpace;
    valSpace = maxValSpace;

    allocateMapVariables();

    startMessageThread();
    MPI_Barrier(MPI_COMM_WORLD);
    mapTimer.start();
    enqueueAllChunks();

    mapTimer.stop();
    mapPostTimer.start();
    binningTimer.start();
    partitionCheckSends(true);
    collectVariablesFromMessageAndKill();
    binningTimer.stop();

    mapFreeTimer.start();
    freeMapVariables();
    mapFreeTimer.stop();
    mapPostTimer.stop();
  }
  void PandaMapReduceJob::sort()
  {
    numUniqueKeys = -1;
    if (sorter->canExecuteOnGPU())  sorter->executeOnGPUAsync(keys, vals, keySize / sizeof(int), numUniqueKeys, &keyOffsets, &valOffsets, &numVals);
    else                            sorter->executeOnCPUAsync(keys, vals, valSize / sizeof(int), numUniqueKeys, &keyOffsets, &valOffsets, &numVals);
  }
  void PandaMapReduceJob::reduce()
  {
    reduceTimer.start();

    emitConfigs.clear();
    keyCount.clear();
    configs.clear();
    cudacpp::DeviceProperties * props = cudacpp::DeviceProperties::get(getDeviceNumber());

    kernelStream = memcpyStream = cudacpp::Stream::nullStream;

    getReduceRunParameters();
    // const int maxMem = props->getTotalMemory() - 10 * 1048576;
    // const int maxReq = maxInputKeySpace + maxInputValSpace + maxKeySpace + maxValSpace;
    // const int numBuffers  = std::min(static_cast<int>(chunks.size()), maxMem / maxReq);
    numBuffers = 1;
    delete props;
    props = NULL;

    allocateReduceVariables();

    configs.resize(numBuffers);
    for (unsigned int i = 0; i < configs.size(); ++i)
    {
      configs[i].keySpace   = reinterpret_cast<char * >(gpuKeys) + maxKeySpace * (i % numBuffers);
      configs[i].valueSpace = reinterpret_cast<char * >(gpuVals) + maxValSpace * (i % numBuffers);
    }

    enqueueReductions();

    reduceTimer.stop();

    cudacpp::Runtime::sync();

    freeReduceVariables();
  }
  void PandaMapReduceJob::collectTimings()
  {
    /*
    const char * const descriptions[] =
    {
      "map",
      "bin",
      "sort",
      "reduce",
      "total",
      // "mapfree",
      // "mappost",
      // "fullmap",
      // "fullreduce",
      // "fulltime",
    };
    */
    double times[] =
    {
      mapTimer.getElapsedSeconds(),
      binningTimer.getElapsedSeconds(),
      sortTimer.getElapsedSeconds(),
      reduceTimer.getElapsedSeconds(),
      totalTimer.getElapsedSeconds(),
      // mapFreeTimer.getElapsedSeconds(),
      // mapPostTimer.getElapsedSeconds(),
      // fullMapTimer.getElapsedSeconds(),
      // fullReduceTimer.getElapsedSeconds(),
      // fullTimer.getElapsedSeconds(),
    };
    const int NUM_TIMES = sizeof(times) / sizeof(double);
    double * allTimes = new double[commSize * NUM_TIMES];
    MPI_Gather(times, NUM_TIMES, MPI_DOUBLE, allTimes, NUM_TIMES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (commRank != 0)
    {
      delete [] allTimes;
      return;
    }
    for (int i = 0; i < NUM_TIMES; ++i)
    {
      double min = times[i], max = times[i];
      double sum = 0.0f;
      for (int j = 0; j < commSize; ++j)
      {
        const double f = allTimes[j * NUM_TIMES + i];
        sum += f;
        min = std::min(min, f);
        max = std::max(max, f);
      }
      // printf("%-6s %10.3f %10.3f %10.3f\n", descriptions[i], min, max, sum / static_cast<double>(commSize));
      // printf(" %s=%5.3f", descriptions[i], sum / static_cast<double>(commSize));
      printf(" %5.3f", sum / static_cast<double>(commSize));
    }
    printf("\n");
    fflush(stdout);
    delete [] allTimes;
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

  void PandaMapReduceJob::addInput(Chunk * chunk)
  {
    chunks.push_back(chunk);
	addMapTasks(chunk);
  }//void

  void PandaMapReduceJob::addMapTasks(panda::Chunk *chunk)
  {
	  //chunks.push_back(chunk);
	  void *key = chunk->getKey();
	  int keySize = chunk->getKeySize();
	  void *val = chunk->getVal();
	  int valSize = chunk->getValSize();

	  MapTask *pMapTask = new MapTask(keySize,key,valSize,val);
	  mapTasks.push_back(pMapTask);

  }//void

  void PandaMapReduceJob::execute()
  {

    mapTimer.start();
    binningTimer.start();
    sortTimer.start();
    reduceTimer.start();
    totalTimer.start();

    mapTimer.stop();
    binningTimer.stop();
    sortTimer.stop();
    reduceTimer.stop();
    totalTimer.stop();

    fullTimer.start();
    fullMapTimer.start();

    if (mapper != NULL)
    {
      if (messager        != NULL) messager->init();
      if (partitioner     != NULL) partitioner->init();
      if (partialReducer  != NULL) partialReducer->init();
      if (combiner        != NULL) combiner->init();

      mapper->init();
      MPI_Barrier(MPI_COMM_WORLD);
      totalTimer.start();
      map();

	  //Panda Process
	  InitMapTasks();
	  InitPandaGPUMapReduce(this->pGPUContext);
	  StartPandaGPUMapTasks(this->pGPUContext);
	  //Panda Code

	  //if(d_g_state->local_combiner){
	  StartGPUCombiner(this->pGPUContext);
	  //}

      mapper->finalize();

      if (messager          != NULL) messager->finalize();
      if (partitioner     != NULL) partitioner->finalize();
      if (partialReducer  != NULL) partialReducer->finalize();
      if (combiner        != NULL) combiner->init();
    }

    fullMapTimer.stop();

	double t3 = PandaTimer();

    sortTimer.start();
    if (sorter != NULL)
    {
      sorter->init();
      sort();
      sorter->finalize();
	  StartGPUShuffle(this->pGPUContext);
    }//if
    sortTimer.stop();

	//TODO
	//across mpi process 
	//set it within MapReduceJob
	//panda_node_context *pnc = new panda_node_context;// = CreatePandaContext();
	PandaShuffleMergeGPU(this->pNodeContext, this->pGPUContext);
	
	//
	//Job Scheduler Plan
	//

	int start_row_id = 0;
	int end_row_id = this->pNodeContext->sorted_keyvals_arr_len;
	AddReduceTask4GPU(this->pGPUContext, this->pNodeContext, start_row_id, end_row_id );

    fullReduceTimer.start();
    if (reducer != NULL && numUniqueKeys > 0)
    {
      reducer->init();
      reduce();
      totalTimer.stop();
      reducer->finalize();
    }
    fullReduceTimer.stop();
    fullTimer.stop();

    collectTimings();
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
