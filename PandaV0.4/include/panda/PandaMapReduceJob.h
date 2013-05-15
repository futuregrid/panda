#ifndef __PANDA_MAPREDUCEJOB_H__
#define __PANDA_MAPREDUCEJOB_H__

#include <panda/Message.h>
#include <panda/EmitConfiguration.h>
#include <panda/PandaGPUConfig.h>
#include <panda/MapReduceJob.h>
#include <cudacpp/Stream.h>
#include <oscpp/Thread.h>
#include <oscpp/Timer.h>
#include <vector>
#include "Panda.h"


namespace panda
{

  class Chunk;
  class MapTask;
  class ReduceTask;
  class EmitConfiguration;

  class PandaMapReduceJob : public MapReduceJob
  {

    protected:
      std::vector<Chunk * > chunks;
	  std::vector<MapTask *> mapTasks;
	  std::vector<ReduceTask *> reduceTasks;

      void * keys;
      void * vals;
      int keySize, keySpace;
      int valSize, valSpace;
      int numUniqueKeys;
      int * keyOffsets;
      int * valOffsets;
      int * numVals;

      oscpp::Thread * MessageThread;

      cudacpp::Stream * kernelStream, * memcpyStream;

      // variables for both map and reduce
      void * cpuKeys, * cpuVals, * gpuKeys, * gpuVals;
      int maxStaticMem, maxKeySpace, maxValSpace, numBuffers;
      std::vector<EmitConfiguration> emitConfigs;
	  
	  panda_gpu_context *pGPUContext;
	  //panda_cpu_context *pCPUContext;
	  panda_node_context *pNodeContext;
	  panda_runtime_context *pRuntimeContext;

      //map variables
      void * gpuStaticMems;
      int * cpuKeyOffsets, * cpuValOffsets, * gpuKeyOffsets, * gpuValOffsets;
      int * cpuKeyCounts, * cpuValCounts, * gpuKeyCounts, * gpuValCounts;
      bool accumMap;
      bool syncPartSends;
      std::vector<void * > savedKeys, savedVals;
      std::vector<int> keyAndValCount;
      std::vector<oscpp::AsyncIORequest * > sendReqs;

      // reduce variables
      std::vector<PandaGPUConfig> configs;
      std::vector<int> keyCount;
      bool accumReduce;
      int maxInputKeySpace, maxInputValSpace, maxInputValOffsetSpace, maxInputNumValsSpace;
      void * gpuInputKeys, * gpuInputVals;
      int  * gpuInputValOffsets, * gpuInputValCounts;

      // timing variables
      oscpp::Timer fullMapTimer, fullReduceTimer, fullTimer;
      oscpp::Timer mapPostTimer, mapFreeTimer;
      oscpp::Timer mapTimer;
      oscpp::Timer binningTimer;
      oscpp::Timer sortTimer;
      oscpp::Timer reduceTimer;
      oscpp::Timer totalTimer;

      virtual void determineMaximumSpaceRequirements();
      virtual void allocateMapVariables();
      virtual void freeMapVariables();
      virtual void startMessageThread();
      virtual void mapChunkExecute(const unsigned int chunkIndex,
                                   PandaGPUConfig & config,
                                   void * const memPool);
      virtual void mapChunkMemcpy(const unsigned int chunkIndex,
                                  const void * const gpuKeySpace,
                                  const void * const gpuValueSpace);
      virtual void mapChunkPartition(const unsigned int chunkIndex,
                                     void * const memPool,
                                     PandaGPUConfig & config);
      virtual void queueChunk(const unsigned int chunkIndex);
      virtual void partitionSubDoGPU(void * const memPool,
                                     void * const keySpace,
                                     void * const valueSpace,
                                     const int numKeys,
                                     const int singleKeySize,
                                     const int singleValSize);
      virtual void partitionSubDoNullPartitioner(const int numKeys);
      virtual void partitionSubSendData(const int singleKeySize, const int singleValSize);
      virtual void partitionCheckSends(const bool sync);

      virtual void partitionSub(void * const memPool,
                                void * const keySpace,
                                void * const valueSpace,
                                const int numKeys,
                                const int singleKeySize,
                                const int singleValSize);
      virtual void partitionChunk(const unsigned int chunkIndex);
      virtual void saveChunk(const unsigned int chunkIndex);
      virtual void combine();
      virtual void globalPartition();
      virtual void enqueueAllChunks();
      virtual void collectVariablesFromMessageAndKill();

      virtual PandaGPUConfig & getReduceConfig(const int index);
      virtual void allocateReduceVariables();
      virtual void freeReduceVariables();
      virtual void getReduceRunParameters();
      virtual void copyReduceInput(const int index, const int keysSoFar);
      virtual void executeReduce(const int index);
      virtual void copyReduceOutput(const int index, const int keysSoFar);
      virtual void enqueueReductions();
	  
	  virtual void InitPandaRuntime();
	  virtual void InitPandaGPUMapReduce();
	  virtual void StartPandaGPUCombiner();
	  virtual void StartPandaSortGPUResults();
	  virtual void StartPandaLocalMergeGPUOutput();
	  virtual void StartPandaGlobalPartition();
	  virtual void StartPandaDoPartitionOnCPU();

	  virtual void StartPandaAddReduceTask4GPU(int start_row_id, int end_row_id);
	  virtual int  StartPandaGPUMapTasks();
	  virtual int  StartPandaGPUReduceTasks();
	  virtual void PandaPartitionCheckSends(const bool sync);
	  virtual void StartPandaPartitionSubSendData();

	  //virtual void StartPandaRuntimeMerge();
	  virtual int GetHash(const char* Key, int KeySize, int commRank );
	  virtual void AddKeyValue2Bucket(int bucketId, const char*key, int keySize, const char*val, int valSize);

      virtual void map();
      virtual void sort();
      virtual void reduce();
      virtual void collectTimings();

    public:
      PandaMapReduceJob(int & argc,
                            char **& argv,
                            const bool accumulateMapResults     = false,
                            const bool accumulateReduceResults  = false,
                            const bool syncOnPartitionSends     = true);
      ~PandaMapReduceJob();

      virtual void addInput(panda::Chunk * chunk);
	  virtual void addMapTasks(panda::Chunk *chunk);
	  
      virtual void execute();
  };
}

#endif
