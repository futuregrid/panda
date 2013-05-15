#include "WCMapper.h"
#include <panda/PreLoadedPandaChunk.h>
#include <cudacpp/Runtime.h>
#include <cstring>
#include <cstdio>

void wcMapperExecute(const int dataSize,
                         void * const gpuMemoryForChunk,
                         PandaGPUConfig & pandaGPUConfig,
                         cudaStream_t & stream);

WCMapper::WCMapper()
{
	//this->dataSize = dataSize;
}

WCMapper::~WCMapper()
{
  
}

panda::EmitConfiguration WCMapper::getEmitConfiguration(panda::Chunk * const chunk) const
{
  const int keySize = sizeof(int);
  const int valSize = sizeof(int);

  dim3 gridSize(1, 1, 1);
  dim3 blockSize(1, 1, 1);
  return panda::EmitConfiguration::createGridConfiguration(keySize,
                                                          valSize,
                                                          gridSize, blockSize, 1,
                                                          keySize,  valSize);
}

bool WCMapper::canExecuteOnGPU() const
{
  return true;
}

bool WCMapper::canExecuteOnCPU() const
{
  return false;
}

void WCMapper::init()
{
  //accumulatedCenters  = reinterpret_cast<float * >(cudacpp::Runtime::mallocDevice(sizeof(float) * numCenters * NUM_BLOCKS * numDims));
  //accumulatedTotals   = reinterpret_cast<int   * >(cudacpp::Runtime::mallocDevice(sizeof(int)   * numCenters * NUM_BLOCKS));
  //gpuCenters          = reinterpret_cast<float * >(cudacpp::Runtime::mallocDevice(sizeof(float) * numDims * numCenters));
  //cudacpp::Runtime::memset(accumulatedCenters, 0, sizeof(float) * numCenters * NUM_BLOCKS * numDims);
  //cudacpp::Runtime::memset(accumulatedTotals,  0, sizeof(float) * numCenters * NUM_BLOCKS);
  //cudacpp::Runtime::memcpyHtoD(gpuCenters, centers, sizeof(float) * numCenters * numDims);
}

void WCMapper::finalize()
{
//  cudacpp::Runtime::free(accumulatedCenters);
//  cudacpp::Runtime::free(accumulatedTotals);
//  cudacpp::Runtime::free(gpuCenters);
}

void WCMapper::executeOnGPUAsync(panda::Chunk * const chunk, PandaGPUConfig & pandaGPUConfig, void * const gpuMemoryForChunk,
                                     cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream)
{
  panda::PreLoadedPandaChunk * fsChunk = dynamic_cast<panda::PreLoadedPandaChunk * >(chunk);

  wcMapperExecute(fsChunk->getMemoryRequirementsOnGPU(),
                      gpuMemoryForChunk,
                      pandaGPUConfig,
                      kernelStream->getHandle());
}

void WCMapper::executeOnCPUAsync(panda::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig)
{

}//void

