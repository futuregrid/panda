#ifndef __PANDA_MAPREDUCEJOB_H__
#define __PANDA_MAPREDUCEJOB_H__

#include <panda/Message.h>
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

  class PandaMapReduceJob : public MapReduceJob
  {

    protected:
	  panda_gpu_card_context	*pGPUCardContext;
	  panda_gpu_context			*pGPUContext;
	  panda_cpu_context			*pCPUContext;
	  panda_node_context		*pNodeContext;
	  panda_runtime_context		*pRuntimeContext;

	  bool enableCPU;
	  bool enableGPU;
	  bool enableGPUCard;

      std::vector<Chunk * > chunks;

	  std::vector<MapTask*>		gpuCardMapTasks;
	  std::vector<MapTask *>	gpuMapTasks;
	  std::vector<MapTask *>	cpuMapTasks;
	  std::vector<ReduceTask *> reduceTasks;

      oscpp::Thread		* pMessageThread;
      cudacpp::Stream	* kernelStream, * memcpyStream;

      bool syncPartSends;
      std::vector<oscpp::AsyncIORequest * > sendReqs;
      std::vector<int> keyCount;
  
	  void PandaInitMapReduceOnCPU();
	  void PandaInitRuntime();
	  void PandaInitMapReduceOnGPU();
	  void PandaInitMapReduceOnGPUCard();

	  void PandaLaunchMessageThread();
      void PandaLaunchPartitionCheckSends(const bool sync);
	  void PandaLaunchCombinerOnGPU();
	  void PandaLaunchCombinerOnGPUCard();
	  void PandaLaunchSortResultsOnGPU();
	  void PandaLaunchLocalMergeOutputOnGPU();
	  void PandaLaunchGlobalPartition();
	  void PandaLaunchPartitionOnCPU();
	  void PandaLaunchSortBucket();
	  void PandaLaunchSortResultsOnCPU();
	  void PandaLaunchSortResultsOnGPUCard();

	  void PandaLaunchCopyRecvedBucketToCPU(int start_task_id, int end_task_id);
	  //void StartPandaAssignReduceTaskToGPU(int start_task_id, int end_task_id);
	  //void StartPandaAssignReduceTaskToGPUCard(int start_task_id, int end_task_id);

	  int  PandaLaunchMapTasksOnGPUCard();
	  int  PandaLaunchMapTasksOnGPUHost();
	  int  PandaLaunchMapTasksOnCPU();
	  int  PandaLaunchReduceTasksOnGPU();
	  int  PandaLaunchReduceTasksOnGPUCard();

	  void PandaPartitionCheckSends(const bool sync);
	  void PandaLaunchPartitionSubSendData();
		
	  void PandaLaunchCopyRecvedBucketToGPU(int start_task_id, int end_task_id);
	  void PandaLaunchCopyRecvedBucketToGPUCard(int start_task_id, int end_task_id);
		
	  void PandaLaunchExitMessager();
		
	  void PandaLaunchCombinerOnCPU();
	  void PandaLaunchReduceTasksOnCPU();
		
	  int  GetHash(const char* Key, int KeySize, int commRank );
	  void PandaAddKeyValue2Bucket(int bucketId, const char*key, int keySize, const char*val, int valSize);
		
	  void partitionSub(void * const memPool,
                                void * const keySpace,
                                void * const valueSpace,
                                const int numKeys,
                                const int singleKeySize,
                                const int singleValSize);

      void partitionChunk(const unsigned int chunkIndex);
      //virtual void collectVariablesFromMessageAndKill();

    public:
      PandaMapReduceJob(	int & argc,
                            char **& argv,
                            const bool accumulateMapResults     = false,
                            const bool accumulateReduceResults  = false,
                            const bool syncOnPartitionSends     = true);
      ~PandaMapReduceJob();

      void addInput(panda::Chunk * chunk);
	  void addCPUMapTasks(Chunk *chunk) ;
	  void addGPUMapTasks(Chunk *chunk) ;
	  void addGPUCardMapTasks(Chunk *chunk);

	  bool getEnableCPU()		{return enableCPU;}
	  bool getEnableGPU()		{return enableGPU;}
	  bool getEnableGPUCard()	{return enableGPUCard;}

	  void setEnableCPU(bool b)		{enableCPU = b;}
	  void setEnableGPU(bool b)		{enableGPU = b;}
	  void setEnableGPUCard(bool b)	{enableGPUCard = b;}
      void execute();

  };
}

#endif
