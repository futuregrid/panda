#ifndef __PANDA_PANDAGPUATOMICFUNCTIONS_H__
#define __PANDA_PANDAGPUATOMICFUNCTIONS_H__

#include <panda/PandaGPUFunctions.h>

template <typename Key, typename Value>
__device__ void gpmrAtomicEmitKeyValRegister(const int outputNumber, const Key & key, const Value & value)
{
}

template <typename Key, typename Value>
__device__ void gpmrAtomicEmitKeyValShared(const int outputNumber, const Key * const keyDataForBlock, const Value * const valueDataForBlock)
{
}

template <typename Key, typename Value>
__device__ void gpmrAtomicEmitKeyValGlobal(const int outputNumber, const Key * const keyDataForBlock, const Value * const valueDataForBlock)
{
}

#endif
