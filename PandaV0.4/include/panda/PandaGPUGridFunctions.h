#ifndef __GPMR_GPMRGPUGRIDFUNCTIONS_H__
#define __GPMR_GPMRGPUGRIDFUNCTIONS_H__

#include <panda/PandaGPUFunctions.h>

#include <cstdio>

template <typename Key, typename Value>
__device__ void gpmrGridEmitKeyValRegister(PandaGPUConfig * const config, const int outputNumber, const Key & key, const Value & value)
{
}

template <typename Key, typename Value>
__device__ void gpmrGridEmitKeyValShared(PandaGPUConfig * const config,
                                         const int blockOffset,
                                         const int outputNumber,
                                         const Key   * const keyDataForBlock,
                                         const Value * const valueDataForBlock)
{
  __shared__ const int * begin;
  __shared__ const int * end;
  int * output;

  const int numThreadsInBlock = blockDim.x * blockDim.y * blockDim.z;                                           // num of threads in the block.
  const int numBlocksInGrid   = gridDim.x * gridDim.y;                                                          // num blocks in the grid.
  const int pitch             = numThreadsInBlock;                                                              // how many copies can be done in one pass.
  const int offset            = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; // the offset within the block (local thread index).
  const int gridOffset        = (blockIdx.x + blockOffset + blockIdx.y * gridDim.x) * numThreadsInBlock;        // the offset within the grid (local block index);
  const int outputBase        = numThreadsInBlock * numBlocksInGrid * outputNumber;                             // global output offset for the output/iteration number.
  const int outputOffset      = outputBase + gridOffset + offset;                                               // offset within the iteration

  begin   = reinterpret_cast<const int * >(keyDataForBlock) + offset;
  end     = reinterpret_cast<const int * >(keyDataForBlock + numThreadsInBlock);
  output  = reinterpret_cast<      int * >(config->keySpace) + outputOffset;

  while (begin < end)
  {
    *output  = *begin;
     output +=  pitch;
     begin  +=  pitch;
  }

  begin   = reinterpret_cast<const int * >(valueDataForBlock) + offset;
  end     = reinterpret_cast<const int * >(valueDataForBlock + numThreadsInBlock);
  output  = reinterpret_cast<      int * >(config->valueSpace) + outputOffset;

  while (begin < end)
  {
    *output  = *begin;
     output +=  pitch;
     begin  +=  pitch;
  }

}

template <typename Key, typename Value>
__device__ void gpmrGridEmitKeyValGlobal(PandaGPUConfig * const config, const int outputNumber, const Key * const keyDataForBlock, const Value * const valueDataForBlock)
{
  const int * begin;
  const int * end;
  int * output;

  const int numThreadsInBlock = blockDim.x * blockDim.y * blockDim.z;                                           // num of threads in the block.
  const int numBlocksInGrid   = gridDim.x * gridDim.y;                                                          // num blocks in the grid.
  const int pitch             = numThreadsInBlock;                                                              // how many copies can be done in one pass.
  const int offset            = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; // the offset within the block (local thread idx).
  const int outputBase        = numThreadsInBlock * numBlocksInGrid * outputNumber;                             // global output offset for the output/iteration number.
  const int outputOffset      = outputBase + offset;                                                            // offset within the iteration

  begin = reinterpret_cast<const int * >(keyDataForBlock) + offset;
  end   = reinterpret_cast<const int * >(keyDataForBlock + numThreadsInBlock);
  output = reinterpret_cast<int * >(reinterpret_cast<char * >(config->keySpace) + (outputBase + outputOffset) * sizeof(Key));

  while (begin < end)
  {
    *output  = *begin;
     output +=  pitch;
     begin  +=  pitch;
  }

  begin = reinterpret_cast<const int * >(valueDataForBlock) + offset;
  end   = reinterpret_cast<const int * >(valueDataForBlock + numThreadsInBlock);
  output = reinterpret_cast<int * >(reinterpret_cast<char * >(config->valueSpace) + (outputBase + outputOffset) * sizeof(Value));

  while (begin < end)
  {
    *output  = *begin;
     output +=  pitch;
     begin  +=  pitch;
  }
}

#endif
