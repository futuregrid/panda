#ifndef __GPMR_GPMRGPUTHREADFUNCTIONS_H__
#define __GPMR_GPMRGPUTHREADFUNCTIONS_H__

#include <panda/PandaGPUFunctions.h>

template <typename Key, typename Value>
__device__ void pandaThreadEmitKeyValShared(const int outputNumber, const Key & key, const Value & value)
{
}

template <typename Key, typename Value>
__device__ void pandaThreadEmitKeyValShared(const int outputNumber, const Key * const keyDataForBlock, const Value * const valueDataForBlock)
{
}

template <typename Key, typename Value>
__device__ void pandaThreadEmitKeyValGlobal(const int outputNumber, const Key * const keyDataForBlock, const Value * const valueDataForBlock)
{
}

#endif
