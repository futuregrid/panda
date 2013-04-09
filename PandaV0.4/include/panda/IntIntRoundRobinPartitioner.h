#ifndef __INTINTROUNDROBINPARTITIONER_H__
#define __INTINTROUNDROBINPARTITIONER_H__

#include <panda/Partitioner.h>

namespace panda
{
  class IntIntRoundRobinPartitioner : public panda::Partitioner
  {
    protected:
      int commSize;
    public:
      IntIntRoundRobinPartitioner();
      virtual ~IntIntRoundRobinPartitioner();

      virtual bool canExecuteOnGPU() const;
      virtual bool canExecuteOnCPU() const;
      virtual int  getMemoryRequirementsOnGPU(panda::EmitConfiguration & emitConfig) const;
      virtual void init();
      virtual void finalize();
      virtual void executeOnGPUAsync(const int numKeys,
                                     const int singleKeySize, const int singleValSize,
                                     void * const gpuKeys,    void * const gpuVals,
                                     int * gpuKeyOffsets,     int * gpuValOffsets,
                                     int * gpuKeyCounts,      int * gpuValCounts,
                                     void * const gpuMemory,
                                     cudacpp::Stream * kernelStream);
      virtual void executeOnCPUAsync(GPMRCPUConfig gpmrCPUConfig, int * keyOffsets, int * valOffsets);
  };
}

#endif
