#ifndef __PANDA_FIXEDSIZERANGEPARTITIONER_H__
#define __PANDA_FIXEDSIZERANGEPARTITIONER_H__

#include <panda/Partitioner.h>

namespace panda
{
  class EmitConfiguration;
  class PandaRangePartitioner : public Partitioner
  {
	  
    protected:
      int rangeBegin, rangeEnd;
      int commSize;
    public:
      PandaRangePartitioner(const int pRangeBegin, const int pRangeEnd);
      virtual ~PandaRangePartitioner();

      virtual bool canExecuteOnGPU() const;
      virtual bool canExecuteOnCPU() const;
      virtual int  getMemoryRequirementsOnGPU(panda::EmitConfiguration & emitConfig) const;
      virtual void init();
      virtual void finalize();
      virtual void executeOnGPUAsync(PandaGPUConfig pandaGPUConfig,
                                     int * gpuKeyOffsets, int * gpuValOffsets,
                                     int * gpuKeyCounts,  int * gpuValCounts,
                                     panda::EmitConfiguration & emitConfig,
                                     void * const gpuMemory,
                                     cudacpp::Stream * kernelStream);
      virtual void executeOnCPUAsync(GPMRCPUConfig gpmrCPUConfig, int * keyOffsets, int * valOffsets);
  };
}

#include <panda/PandaRangerPartitioner.cxx>

#endif
