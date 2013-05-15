#ifndef __GPMR_PARTIALSORTER_H__
#define __GPMR_PARTIALSORTER_H__

#include <panda/PandaCPUConfig.h>
#include <panda/PandaGPUConfig.h>

namespace panda
{
  class PartialSorter
  {
    public:
      PartialSorter();
      virtual ~PartialSorter();

      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual void executeOnGPUAsync(PandaGPUConfig * const pandaGPUConfig) = 0;
      virtual void executeOnCPUAsync(GPMRCPUConfig * const gpmrCPUConfig) = 0;
  };
}

#endif
