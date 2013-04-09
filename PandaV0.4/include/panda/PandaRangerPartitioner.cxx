#ifndef __GPMR_FIXEDSIZERANGEPARTITIONER_CXX__
#define __GPMR_FIXEDSIZERANGEPARTITIONER_CXX__

#include <panda/PandaRangePartitioner.h>

#include <mpi.h>

namespace panda
{
  class EmitConfiguration;

  template <typename Key, typename Val> void PandaRangePartitionerZeroCountsAndIndices(const int commSize, int * gpuBucketCounts, int * gpuBucketIndices, cudaStream_t * stream);
  template <typename Key, typename Val> void PandaRangePartitionerCount               (const int commSize, const int numKeys, Key * gpuKeys, int * gpuBucketCounts, cudaStream_t * stream);
//  template <typename Key, typename Value> void PandaRangePartitionerCount     (const int commSize, const int numKeys, Key * gpuKeys, Value * gpuVals, Key * gpuTempKeySpace, Vakye * gpuTempValSpace, int * gpuKeyOffsets, int * gpuBucketIndices, cudaStream_t * stream);
//  template <typename Key, typename Value> void PandaRangePartitionerMoveFromTemp        (const int commSize, const int numKeys, Key * gpuKeys, Value * gpuVals, Key * gpuTempKeySpace, Vakye * gpuTempValSpace, cudaStream_t * stream);
  template <typename Key, typename Val> void PandaRangePartitionerSetKeyAndValCounts  (const int commSize, int * gpuKeyCounts, int * gpuValCounts, cudaStream_t * stream);
  template <typename Key, typename Val> void PandaRangePartitionerSetKeyAndValOffsets (const int commSize, const int * const gpuKeyCounts, int * const gpuKeyOffsets, int * const gpuValOffsets, cudaStream_t * stream);

  //template <typename Key, typename Value>
  /*PandaRangePartitioner<Key, Value>::PandaRangePartitioner(const Key & pRangeBegin, const Key & pRangeEnd)
  {
    rangeBegin = pRangeBegin;
    rangeEnd   = pRangeEnd;
  }*/
  //template <typename Key, typename Value>
  //PandaRangePartitioner<Key, Value>::~PandaRangePartitioner()
  //{
  //}
  
  template <typename Key, typename Value>
  bool PandaRangePartitioner<Key, Value>::canExecuteOnGPU() const
  {
    return true;
  }//bool

  template <typename Key, typename Value>
  bool PandaRangePartitioner<Key, Value>::canExecuteOnCPU() const
  {
    return false;
  }//

  template <typename Key, typename Value>
  int  PandaRangePartitioner<Key, Value>::getMemoryRequirementsOnGPU(panda::EmitConfiguration & emitConfig) const
  {
    return emitConfig.getKeySpace() + emitConfig.getValueSpace();
  }

  template <typename Key, typename Value>
  void PandaRangePartitioner<Key, Value>::init()
  {
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  }

  template <typename Key, typename Value>
  void PandaRangePartitioner<Key, Value>::finalize()
  {
  }

  template <typename Key, typename Value>
  void PandaRangePartitioner<Key, Value>::executeOnGPUAsync(PandaGPUConfig pandaGPUConfig,
                                                                int * gpuKeyOffsets, int * gpuValOffsets,
                                                                int * gpuKeyCounts,  int * gpuValCounts,
                                                                panda::EmitConfiguration & emitConfig,
                                                                void * const gpuMemory,
                                                                cudacpp::Stream * kernelStream)
  {
    const int numKeys = pandaGPUConfig.emitInfo.grid.numThreads * pandaGPUConfig.emitInfo.grid.emitsPerThread;
    Key   * gpuTempKeySpace   = reinterpret_cast<Key * >(gpuMemory);
    Value * gpuTempValSpace   = reinterpret_cast<Value * >(gpuTempKeySpace + emitConfig.getIndexCount());
    int * gpuBucketCounts   = gpuKeyCounts;
    int * gpuBucketIndices  = gpuValCounts;
    Key   * gpuKeys           = reinterpret_cast<Key   * >(pandaGPUConfig.keySpace);
    Value * gpuVals           = reinterpret_cast<Value * >(pandaGPUConfig.valueSpace);

    PandaRangePartitionerZeroCountsAndIndices(commSize, gpuBucketCounts, gpuBucketIndices, &kernelStream->getHandle());
    PandaRangePartitionerCount               (commSize, numKeys, gpuKeys, gpuBucketCounts, &kernelStream->getHandle());
    PandaRangePartitionerSetKeyAndValOffsets (commSize, gpuKeyCounts, gpuKeyOffsets, gpuValOffsets, &kernelStream->getHandle());
    PandaRangePartitionerCount     (commSize, numKeys, gpuKeys, gpuVals, gpuTempKeySpace, gpuTempValSpace, gpuKeyOffsets, gpuBucketIndices, &kernelStream->getHandle());
    PandaRangePartitionerMoveFromTemp        (commSize, numKeys, gpuKeys, gpuVals, gpuTempKeySpace, gpuTempValSpace, &kernelStream->getHandle());
    PandaRangePartitionerSetKeyAndValCounts  (commSize, gpuKeyCounts, gpuValCounts, &kernelStream->getHandle());
    PandaRangePartitionerSetKeyAndValOffsets (commSize, gpuKeyCounts, gpuKeyOffsets, gpuValOffsets, &kernelStream->getHandle());
  }

  // even though this isn't supported, we need to add it. otherwise the compiler
  // will give lots of errors.
  template <typename Key, typename Value>
  void PandaRangePartitioner<Key, Value>::executeOnCPUAsync(GPMRCPUConfig gpmrCPUConfig, int * keyOffsets, int * valOffsets)
  {
    const float range = static_cast<float>(rangeEnd - rangeBegin);
    const int numItems = gpmrCPUConfig.emitInfo.grid.numThreads * gpmrCPUConfig.emitInfo.grid.emitsPerThread;
    Key   * keys = reinterpret_cast<Key * >(gpmrCPUConfig.keySpace);
    Value * vals = reinterpret_cast<Value * >(gpmrCPUConfig.valueSpace);

    for (int i = 0; i < commSize; ++i) keyOffsets[i] = valOffsets[i] = 0;

    int * keyCounts = new int[commSize];
    int * valCounts = new int[commSize];

    memset(keyCounts, 0, sizeof(int) * commSize);
    memset(valCounts, 0, sizeof(int) * commSize);

    for (int i = 0; i < numItems; ++i)
    {
      const float        f     = static_cast<float>(keys[i] - rangeBegin) / range;
      const unsigned int index = static_cast<unsigned int>(f * commSize);
      ++keyCounts[index];
      ++valCounts[index];
    }

    keyOffsets[0] = valOffsets[0] = 0;
    for (int i = 1; i < commSize; ++i)
    {
      keyOffsets[i] = keyOffsets[i - 1] + keyCounts[i - 1];
      valOffsets[i] = valOffsets[i - 1] + valCounts[i - 1];
    }

    Key   * tempKeys = new Key  [numItems];
    Value * tempVals = new Value[numItems];

    memset(keyCounts, 0, sizeof(int) * commSize);
    for (int i = 0; i < numItems; ++i)
    {
      const float        f     = static_cast<float>(keys[i] - rangeBegin) / range;
      const unsigned int index = static_cast<unsigned int>(f * commSize);
      tempKeys[keyOffsets[index] + keyCounts[index]] = keys[i];
      tempVals[keyOffsets[index] + keyCounts[index]] = vals[i];
      keyCounts[index]++;
    }

    memcpy(keys, tempKeys, sizeof(Key)   * numItems);
    memcpy(vals, tempVals, sizeof(Value) * numItems);

    delete [] tempKeys;
    delete [] tempVals;
  }
}

#endif
