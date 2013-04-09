#ifndef __GPMR_GPMRGPUBLOCKFUNCTIONS_H__
#define __GPMR_GPMRGPUBLOCKFUNCTIONS_H__

#include <panda/PandaGPUFunctions.h>

template <typename Key, typename Value>
__device__ void gpmrBlockEmitKeyValRegister(const int outputNumber, const Key & key, const Value & value)
{
}

template <typename Key, typename Value>
__device__ void gpmrBlockEmitKeyValShared(const int outputNumber, const Key & key, const Value & value)
{
}

template <typename Key, typename Value>
__device__ void gpmrBlockEmitKeyValGlobal(const int outputNumber, const Key & key, const Value & value)
{
}

#endif
