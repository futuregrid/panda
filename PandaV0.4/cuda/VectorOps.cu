#ifndef __VECTOROPS_CU__
#define __VECTOROPS_CU__

#include <math_functions.h>
#include <panda/VectorOps.h>

inline __device__ int3 imin(const int3 & i0, const int3 & i1) { return int3_ctor(min(i0.x, i1.x), min(i0.y, i1.y), min(i0.z, i1.z)); }
inline __device__ int3 imax(const int3 & i0, const int3 & i1) { return int3_ctor(max(i0.x, i1.x), max(i0.y, i1.y), max(i0.z, i1.z)); }

inline __device__ float3 fminf(const float3 & i0, const float3 & i1) { return float3_ctor(fminf(i0.x, i1.x), fminf(i0.y, i1.y), fminf(i0.z, i1.z)); }
inline __device__ float3 fmaxf(const float3 & i0, const float3 & i1) { return float3_ctor(fmaxf(i0.x, i1.x), fmaxf(i0.y, i1.y), fmaxf(i0.z, i1.z)); }

#endif
