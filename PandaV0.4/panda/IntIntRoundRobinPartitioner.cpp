#include <mpi.h>
#include <panda/IntIntRoundRobinPartitioner.h>
#include <panda/EmitConfiguration.h>

void intIntRoundRobinPartitionerExecute(const int commSize,     const int numKeys,
                                        int * const keys,       int * const vals,
                                        int * const tempKeys,   int * const tempVals,
                                        int * const keyCounts,  int * const valCounts,
                                        int * const keyOffsets, int * const valOffsets,
                                        cudaStream_t stream);

namespace panda
{
  IntIntRoundRobinPartitioner::IntIntRoundRobinPartitioner()
  {
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  }
  IntIntRoundRobinPartitioner::~IntIntRoundRobinPartitioner()
  {
  }

  bool IntIntRoundRobinPartitioner::canExecuteOnGPU() const
  {
    return true;
  }
  bool IntIntRoundRobinPartitioner::canExecuteOnCPU() const
  {
    return false;
  }
  int IntIntRoundRobinPartitioner::getMemoryRequirementsOnGPU(panda::EmitConfiguration & emitConfig) const
  {
    return emitConfig.getIndexCount() * sizeof(int) * 2;
  }
  void IntIntRoundRobinPartitioner::init()
  {
  }
  void IntIntRoundRobinPartitioner::finalize()
  {
  }
  void IntIntRoundRobinPartitioner::executeOnGPUAsync(const int numKeys,
                                                      const int singleKeySize, const int singleValSize,
                                                      void * const gpuKeys,    void * const gpuVals,
                                                      int * gpuKeyOffsets,     int * gpuValOffsets,
                                                      int * gpuKeyCounts,      int * gpuValCounts,
                                                      void * const gpuMemory,
                                                      cudacpp::Stream * kernelStream)
  {
    /*
      basic algorithm:

      count number of keys in each bucket
      scan bucket counts into bucketIndices
      move data from keys and vals to tempKeys and tempVals
    */

    int * const ikeys   = reinterpret_cast<int * >(gpuKeys);
    int * const ivals   = reinterpret_cast<int * >(gpuVals);
    int * const mem     = reinterpret_cast<int * >(gpuMemory);
    int * tempKeys      = mem;
    int * tempVals      = tempKeys      + numKeys;
    // int * bucketCounts  = tempVals      + numKeys;
    // int * bucketIndices = bucketCounts  + commSize;

    intIntRoundRobinPartitionerExecute(commSize,      numKeys,
                                       ikeys,         ivals,
                                       tempKeys,      tempVals,
                                       gpuKeyCounts,  gpuValCounts,
                                       gpuKeyOffsets, gpuValOffsets,
                                       kernelStream->getHandle());
  }
  void IntIntRoundRobinPartitioner::executeOnCPUAsync(GPMRCPUConfig gpmrCPUConfig, int * keyOffsets, int * valOffsets)
  {
  }
}
