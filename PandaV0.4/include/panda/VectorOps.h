#ifndef __VECTOROPS_H__
#define __VECTOROPS_H__

#ifdef _WIN32
  #pragma warning(disable:4365)
  #pragma warning(disable:4505)
  #pragma warning(disable:4514)
  #pragma warning(disable:4619)
  #pragma warning(disable:4668)
#endif

#include <builtin_types.h>
#include <cmath>

inline __device__ __host__ int3 int3_ctor(const int x, const int y, const int z)  { int3 i3 = { x, y, z }; return i3; }
inline __device__ __host__ int3 int3_ctor(const int x)                            { int3 i3 = { x, x, x }; return i3; }

inline __device__ __host__ float3 float3_ctor(const float x, const float y, const float z)  { float3 ret = { x, y, z }; return ret; }
inline __device__ __host__ float3 float3_ctor(const float x)                                { float3 ret = { x, x, x }; return ret; }
inline __device__ __host__ float3 float3_ctor(const int3 & i)                               { float3 ret = { static_cast<float>(i.x), static_cast<float>(i.y), static_cast<float>(i.z) }; return ret; }
inline __device__ __host__ float3 float3_ctor(const int x, const int y, const int z)        { float3 ret = { static_cast<float>(x),   static_cast<float>(y),   static_cast<float>(z)   }; return ret; }

inline __device__ __host__ int3 operator + (const int3 & lhs, const int3 & rhs) { int3 ret = { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z }; return ret; }
inline __device__ __host__ int3 operator - (const int3 & lhs, const int3 & rhs) { int3 ret = { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z }; return ret; }
inline __device__ __host__ int3 operator * (const int3 & lhs, const int3 & rhs) { int3 ret = { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z }; return ret; }
inline __device__ __host__ int3 operator / (const int3 & lhs, const int3 & rhs) { int3 ret = { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z }; return ret; }

inline __device__ __host__ int3 & operator += (int3 & lhs, const int3 & rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }
inline __device__ __host__ int3 & operator -= (int3 & lhs, const int3 & rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }
inline __device__ __host__ int3 & operator *= (int3 & lhs, const int3 & rhs) { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs; }
inline __device__ __host__ int3 & operator /= (int3 & lhs, const int3 & rhs) { lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; return lhs; }

inline __device__ __host__ int3 operator * (const int3 & lhs, const int rhs) { int3 ret = { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs }; return ret; }
inline __device__ __host__ int3 operator / (const int3 & lhs, const int rhs) { return int3_ctor(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs); }

inline __device__ __host__ int3 operator * (const int lhs, const int3 & rhs) { return rhs * lhs; }
inline __device__ __host__ int3 operator / (const int lhs, const int3 & rhs) { int3 ret = { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z }; return ret; }

inline __device__ __host__ int3 operator *= (int3 & lhs, const int rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; return lhs; }
inline __device__ __host__ int3 operator /= (int3 & lhs, const int rhs) { lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; return lhs; }

inline __device__ __host__ int  imin(const int i0, const int i1) { return i0 <= i1 ? i0 : i1; }
inline __device__ __host__ int  imax(const int i0, const int i1) { return i0 >= i1 ? i0 : i1; }

__host__ __device__ int3 imin(const int3 & i0, const int3 & i1);
__host__ __device__ int3 imax(const int3 & i0, const int3 & i1);

inline __device__ __host__ float3 operator + (const float3 & lhs, const float3 & rhs) { float3 ret = { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z }; return ret; }
inline __device__ __host__ float3 operator - (const float3 & lhs, const float3 & rhs) { float3 ret = { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z }; return ret; }
inline __device__ __host__ float3 operator * (const float3 & lhs, const float3 & rhs) { float3 ret = { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z }; return ret; }
inline __device__ __host__ float3 operator / (const float3 & lhs, const float3 & rhs) { float3 ret = { lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z }; return ret; }

inline __device__ __host__ float3 & operator += (float3 & lhs, const float3 & rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }
inline __device__ __host__ float3 & operator -= (float3 & lhs, const float3 & rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }
inline __device__ __host__ float3 & operator *= (float3 & lhs, const float3 & rhs) { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs; }
inline __device__ __host__ float3 & operator /= (float3 & lhs, const float3 & rhs) { lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; return lhs; }

inline __device__ __host__ float3 operator * (const float3 & lhs, const float rhs) { float3 ret = { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs }; return ret; }
inline __device__ __host__ float3 operator / (const float3 & lhs, const float rhs) { return lhs * (1.0f / rhs); }

inline __device__ __host__ float3 operator * (const float lhs, const float3 & rhs) { return rhs * lhs; }
inline __device__ __host__ float3 operator / (const float lhs, const float3 & rhs) { float3 ret = { lhs / rhs.x, lhs / rhs.y, lhs / rhs.z }; return ret; }

inline __device__ __host__ float3 operator *= (float3 & lhs, const float rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; return lhs; }
inline __device__ __host__ float3 operator /= (float3 & lhs, const float rhs) { lhs *= (1.0f / rhs); return lhs; }

inline __device__ __host__ float mag2(const float3 & lhs) { return lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z; }
inline __device__ __host__ float mag(const float3 & lhs) { return sqrtf(mag2(lhs)); }

inline __device__ __host__ float3 norm(const float3 & lhs) { return lhs / mag(lhs); }
inline __device__ __host__ float3 & norm(float3 & lhs, float3 & res) { float m = 1.0f / mag(lhs); res.x = lhs.x * m; res.y = lhs.y * m; res.z = lhs.z * m; return lhs; }
inline __device__ __host__ float3 & normInPlace(float3 & lhs) { return norm(lhs, lhs); }

inline __device__ __host__ float dot(const float3 & lhs, const float3 & rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
inline __device__ __host__ float3 cross(const float3 & lhs, const float3 & rhs)
{
  const float * const lhsp = &lhs.x;
  const float * const rhsp = &rhs.x;
  float3 ret =
  {
    lhsp[1] * rhsp[2] - lhsp[2] * rhsp[1],
    lhsp[2] * rhsp[0] - lhsp[0] * rhsp[2],
    lhsp[0] * rhsp[1] - lhsp[1] * rhsp[0],
  };
  return ret;
}

__host__ __device__ float3 fminf(const float3 & i0, const float3 & i1);
__host__ __device__ float3 fmaxf(const float3 & i0, const float3 & i1);

#endif
