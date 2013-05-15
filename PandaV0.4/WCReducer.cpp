#include <mpi.h>
#include "WCReducer.h"
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <cudpp/cudpp.h>


void wcReducerExecute(const int numKeys,
                          const int   * const numVals,
                          const int   * const oldKeys,
                                int   * const newKeys,
                          const void  * const oldVals,
                                void  * const newVals,
                          cudaStream_t & stream);

WCReducer::WCReducer()
{
 
}
WCReducer::~WCReducer()
{
}

panda::EmitConfiguration WCReducer::getEmitConfiguration(const void * const keys,
                                                            const int * const numVals,
                                                            const int numKeys,
                                                            int & numKeysToProcess)
{
  // not used
  numKeysToProcess = numKeys;
  return panda::EmitConfiguration::createGridConfiguration(numKeys * sizeof(int), numKeys * sizeof(int),
                                                          dim3(numKeys , 1, 1), dim3(1, 1, 1),
                                                          1,
                                                          sizeof(int), sizeof(int));
}
bool WCReducer::canExecuteOnGPU() const
{
  return true;
}
bool WCReducer::canExecuteOnCPU() const
{
  return false;
}
void WCReducer::init()
{
}
void WCReducer::finalize()
{
}

void WCReducer::executeOnGPUAsync(const int numKeys,
                                      const void * const keys,
                                      const void * const vals,
                                      const int * keyOffsets,
                                      const int * valOffsets,
                                      const int * numVals,
                                      PandaGPUConfig & gpuConfig,
                                      cudacpp::Stream * const kernelStream)
{
  wcReducerExecute(numKeys,
                       numVals,
                       reinterpret_cast<const int * >(keys),
                       reinterpret_cast<int * >(gpuConfig.keySpace),
                       vals,
                       gpuConfig.valueSpace,
                       kernelStream->getHandle());
}
