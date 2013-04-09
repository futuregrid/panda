#ifndef __KMEANSREDUCER_H__
#define __KMEANSREDUCER_H__

#include <panda/Reducer.h>

class WCReducer : public panda::Reducer
{
  //protected:

  public:
    WCReducer();
    virtual ~WCReducer();

    virtual panda::EmitConfiguration getEmitConfiguration(const void * const keys,
                                                         const int * const numVals,
                                                         const int numKeys,
                                                         int & numKeysToProcess);
    virtual bool canExecuteOnGPU() const;
    virtual bool canExecuteOnCPU() const;
    virtual void init();
    virtual void finalize();

    virtual void executeOnGPUAsync(const int numKeys,
                                   const void * const keys,
                                   const void * const vals,
                                   const int * const keyOffsets,
                                   const int * const valOffsets,
                                   const int * const numVals,
                                   PandaGPUConfig & gpuConfig,
                                   cudacpp::Stream * const kernelStream);
};

#endif
