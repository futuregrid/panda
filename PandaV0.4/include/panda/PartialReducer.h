#ifndef __GPMR_PARTIALREDUCER_H__
#define __GPMR_PARTIALREDUCER_H__

#include <cudacpp/Stream.h>

namespace panda
{
  class EmitConfiguration;
  class PartialReducer
  {
    public:
      PartialReducer();
      virtual ~PartialReducer();

      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual int  getMemoryRequirementsOnGPU(EmitConfiguration & emitConfig) const = 0;
      virtual void executeOnGPUAsync(EmitConfiguration & config,
                                     void * const gpuKeys,
                                     void * const gpuVals,
                                     void * const gpuMem,
                                     int * const gpuKeyCounts,
                                     int * const gpuKeyOffsets,
                                     int * const gpuValCounts,
                                     int * const gpuValOffsets,
                                     cudacpp::Stream * stream) = 0;
      virtual void executeOnCPUAsync() = 0;
  };
}

#endif
