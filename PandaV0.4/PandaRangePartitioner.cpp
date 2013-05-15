#ifndef __PANDA_FIXEDSIZERANGEPARTITIONER_CXX__
#define __PANDA_FIXEDSIZERANGEPARTITIONER_CXX__

#include <mpi.h>
#include <panda/PandaRangePartitioner.h>
#include <panda/EmitConfiguration.h>



#include <cstring>
namespace panda
{

  void PandaRangePartitionerZeroCountsAndIndices(const int commSize,
                                                         int * gpuBucketCounts,
                                                         int * gpuBucketIndices,
														 cudaStream_t * stream){}
  void PandaRangePartitionerCount(const int commSize,
                                          const int numKeys,
                                          int * gpuKeys,
                                          int * gpuBucketCounts,
                                          cudaStream_t * stream){}
  void PandaRangePartitionerCount(const int commSize,
                                                    const int numKeys,
                                                    int * gpuKeys,
                                                    int * gpuVals,
                                                    int * gpuTempKeySpace,
                                                    int * gpuTempValSpace,
                                                    int * gpuKeyOffsets,
                                                    int * gpuBucketIndices,
                                                    cudaStream_t * stream){}
  void PandaRangePartitionerMoveFromTemp(const int commSize,
                                                 const int numKeys,
                                                 int * gpuKeys,
                                                 int * gpuVals,
                                                 int * gpuTempKeySpace,
                                                 int * gpuTempValSpace,
                                                 cudaStream_t * stream){}
  void PandaRangePartitionerSetKeyAndValCounts(const int commSize,
                                                       int * gpuKeyCounts,
                                                       int * gpuValCounts,
                                                       cudaStream_t * stream){}

  void PandaRangePartitionerSetKeyAndValOffsets(const int commSize,
                                                        const int * const gpuKeyCounts,
                                                        int * const gpuKeyOffsets,
                                                        int * const gpuValOffsets,
                                                        cudaStream_t * stream){}

  PandaRangePartitioner::PandaRangePartitioner(const int pRangeBegin, const int pRangeEnd)
  {
    rangeBegin = pRangeBegin;
    rangeEnd   = pRangeEnd;
  }
  PandaRangePartitioner::~PandaRangePartitioner()
  {
  }

  bool PandaRangePartitioner::canExecuteOnGPU() const
  {
    return true;
  }

  bool PandaRangePartitioner::canExecuteOnCPU() const
  {
    return false;
  }

  int  PandaRangePartitioner::getMemoryRequirementsOnGPU(panda::EmitConfiguration & emitConfig) const
  {
    return emitConfig.getKeySpace() + emitConfig.getValueSpace();
  }

  void PandaRangePartitioner::init()
  {
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  }

  void PandaRangePartitioner::finalize()
  {
  }

  void PandaRangePartitioner::executeOnGPUAsync(PandaGPUConfig pandaGPUConfig,
                                                    int * gpuKeyOffsets, int * gpuValOffsets,
                                                    int * gpuKeyCounts,  int * gpuValCounts,
                                                    panda::EmitConfiguration & emitConfig,
                                                    void * const gpuMemory,
                                                    cudacpp::Stream * kernelStream)
  {
    const int numKeys = pandaGPUConfig.emitInfo.grid.numThreads * pandaGPUConfig.emitInfo.grid.emitsPerThread;
    int * gpuTempKeySpace   = reinterpret_cast<int * >(gpuMemory);
    int * gpuTempValSpace   = gpuTempKeySpace + numKeys;
    int * gpuBucketCounts   = gpuKeyCounts;
    int * gpuBucketIndices  = gpuValCounts;
    int * gpuKeys           = reinterpret_cast<int * >(pandaGPUConfig.keySpace);
    int * gpuVals           = reinterpret_cast<int * >(pandaGPUConfig.valueSpace);

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
  void PandaRangePartitioner::executeOnCPUAsync(GPMRCPUConfig gpmrCPUConfig, int * keyOffsets, int * valOffsets)
  {
    const float range = static_cast<float>(rangeEnd - rangeBegin);
    const int numItems = gpmrCPUConfig.emitInfo.grid.numThreads * gpmrCPUConfig.emitInfo.grid.emitsPerThread;
    int * keys = reinterpret_cast<int * >(gpmrCPUConfig.keySpace);
    int * vals = reinterpret_cast<int * >(gpmrCPUConfig.valueSpace);

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

    int * tempKeys = new int[numItems];
    int * tempVals = new int[numItems];

    memset(keyCounts, 0, sizeof(int) * commSize);
    for (int i = 0; i < numItems; ++i)
    {
      const float        f     = static_cast<float>(keys[i] - rangeBegin) / range;
      const unsigned int index = static_cast<unsigned int>(f * commSize);
      tempKeys[keyOffsets[index] + keyCounts[index]] = keys[i];
      tempVals[keyOffsets[index] + keyCounts[index]] = vals[i];
      keyCounts[index]++;
    }

    memcpy(keys, tempKeys, sizeof(int) * numItems);
    memcpy(vals, tempVals, sizeof(int) * numItems);

    delete [] tempKeys;
    delete [] tempVals;
  }
}

#endif
