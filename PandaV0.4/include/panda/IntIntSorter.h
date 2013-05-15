#ifndef __GPMR_INTINTSORTER_H__
#define __GPMR_INTINTSORTER_H__

#include <panda/Sorter.h>
#include <panda/PandaCPUConfig.h>
#include <panda/PandaGPUConfig.h>

namespace panda
{
  class IntIntSorter : public Sorter
  {
    public:
      IntIntSorter();
      virtual ~IntIntSorter();

      virtual bool canExecuteOnGPU() const;
      virtual bool canExecuteOnCPU() const;
      virtual void init();
      virtual void finalize();
      virtual void executeOnGPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals);
      virtual void executeOnCPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals);
  };
}

#endif
