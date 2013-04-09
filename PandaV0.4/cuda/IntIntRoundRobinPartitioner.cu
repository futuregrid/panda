#include <cudacpp/Runtime.h>
#include <cstdio>

static __device__ void reduce(volatile int * const mem)
{
  syncthreads();
  if (threadIdx.x < 256) mem[threadIdx.x] += mem[threadIdx.x + 256]; syncthreads();
  if (threadIdx.x < 128) mem[threadIdx.x] += mem[threadIdx.x + 128]; syncthreads();
  if (threadIdx.x <  64) mem[threadIdx.x] += mem[threadIdx.x +  64]; syncthreads();
  if (threadIdx.x <  32)
  {
    mem[threadIdx.x] += mem[threadIdx.x +  32];
    mem[threadIdx.x] += mem[threadIdx.x +  16];
    mem[threadIdx.x] += mem[threadIdx.x +   8];
    mem[threadIdx.x] += mem[threadIdx.x +   4];
    mem[threadIdx.x] += mem[threadIdx.x +   2];
    mem[threadIdx.x] += mem[threadIdx.x +   1];
  }
  syncthreads();
}
static __device__ void scan(volatile int * const array)
{
  __shared__ int extra[32];
  int myVal = array[threadIdx.x];
  const int THREADS_PER_WARP      = 16;
  const int THREAD_INDEX_IN_WARP  = (threadIdx.x & 0xF);
  const int WARP_INDEX            =  threadIdx.x >> 4;
  if (THREAD_INDEX_IN_WARP > 0) { array[threadIdx.x] = array[threadIdx.x - 1] + array[threadIdx.x]; }
  if (THREAD_INDEX_IN_WARP > 1) { array[threadIdx.x] = array[threadIdx.x - 2] + array[threadIdx.x]; }
  if (THREAD_INDEX_IN_WARP > 3) { array[threadIdx.x] = array[threadIdx.x - 4] + array[threadIdx.x]; }
  if (THREAD_INDEX_IN_WARP > 7) { array[threadIdx.x] = array[threadIdx.x - 8] + array[threadIdx.x]; }
  if (THREAD_INDEX_IN_WARP == 0) extra[WARP_INDEX] = array[threadIdx.x + THREADS_PER_WARP - 1];
  __syncthreads();
  if (threadIdx.x < 31) { extra[threadIdx.x +  1] = extra[threadIdx.x] + extra[threadIdx.x +  1]; }
  if (threadIdx.x < 30) { extra[threadIdx.x +  2] = extra[threadIdx.x] + extra[threadIdx.x +  2]; }
  if (threadIdx.x < 28) { extra[threadIdx.x +  4] = extra[threadIdx.x] + extra[threadIdx.x +  4]; }
  if (threadIdx.x < 24) { extra[threadIdx.x +  8] = extra[threadIdx.x] + extra[threadIdx.x +  8]; }
  if (threadIdx.x < 16) { extra[threadIdx.x + 16] = extra[threadIdx.x] + extra[threadIdx.x + 16]; }
  __syncthreads();
  if (WARP_INDEX > 0) array[threadIdx.x] = extra[WARP_INDEX - 1] + array[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) array[threadIdx.x] = 0;
  else                  array[threadIdx.x] -= myVal;
  __syncthreads();
}

static __global__ void iirrpCopy(int * const dst, const int * const src, const int count)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
  {
    dst[i] = src[i];
  }
}
static __global__ void iirrpSmallScan(int * const output, const int * const input, const int count)
{
  output[0] = 0;
  for (int i = 1; i < count; ++i) output[i] = output[i - 1] + input[i - 1];
}
static __device__ int iirrpCountBucketItems(const int numBuckets,
                                            const int numKeys,
                                            const int bucket,
                                            const int * const keys)
{
  int count = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numKeys; i += gridDim.x * blockDim.x)
  {
    if ((keys[i] % numBuckets) == bucket) ++count;
  }
  return count;
}



static __global__ void iirrpZero(const int count, int * const arr)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) arr[i] = 0;
}
static __global__ void iirrpCount(const int numBuckets, const int numKeys, int * const keys, int * const keyCounts)
{
  __shared__ volatile int localCounts[512];
  for (int bucket = 0; bucket < numBuckets; ++bucket)
  {
    localCounts[threadIdx.x] = iirrpCountBucketItems(numBuckets, numKeys, bucket, keys);
    reduce(localCounts);
    if (threadIdx.x == 0) atomicAdd(keyCounts + bucket, localCounts[0]);
  }
}
static __global__ void iirrpMove(const int numBuckets, const int numKeys,
                                 int * const dst,      const int * const src,
                                 int * const dst2,     const int * const src2,
                                 int * const baseOffsets)
{
  __shared__ volatile int localCounts[512];
  for (int bucket = 0; bucket < numBuckets; ++bucket)
  {
    int count = iirrpCountBucketItems(numBuckets, numKeys, bucket, src);
    localCounts[threadIdx.x] = count;
    reduce(localCounts);

    if (threadIdx.x == 0) localCounts[0] = atomicAdd(baseOffsets + bucket, localCounts[0]);
    __syncthreads();
    int offset = localCounts[0];

    __syncthreads();
    localCounts[threadIdx.x] = count;
    scan(localCounts);

    offset += localCounts[threadIdx.x];

    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < numKeys; i += gridDim.x * blockDim.x)
    {
      const int bucketID = src[i] % numBuckets;
      if (bucket == bucketID)
      {
        dst [offset] = src [i];
        dst2[offset] = src2[i];
        ++offset;
      }
    }
  }
}

void intIntRoundRobinPartitionerExecute(const int commSize,     const int numKeys,
                                        int * const keys,       int * const vals,
                                        int * const tempKeys,   int * const tempVals,
                                        int * const keyCounts,  int * const valCounts,
                                        int * const keyOffsets, int * const valOffsets,
                                        cudaStream_t stream)
{
  iirrpZero     <<<60, 512, 0, stream>>>(commSize,    keyCounts);
  iirrpCount    <<<60, 512, 0, stream>>>(commSize,    numKeys,    keys,     keyCounts);
  iirrpCopy     <<<60, 512, 0, stream>>>(valCounts,   keyCounts,  commSize);
  iirrpSmallScan<<< 1,   1, 0, stream>>>(keyOffsets,  keyCounts,  commSize);
  iirrpCopy     <<<60, 512, 0, stream>>>(valOffsets,  keyOffsets, commSize);
  iirrpMove     <<<60, 512, 0, stream>>>(commSize,    numKeys,    tempKeys, keys, tempVals, vals, valOffsets);
  iirrpCopy     <<<60, 512, 0, stream>>>(keys,        tempKeys,   numKeys);
  iirrpCopy     <<<60, 512, 0, stream>>>(vals,        tempVals,   numKeys);
  iirrpCopy     <<<60, 512, 0, stream>>>(keyCounts,   valCounts,  commSize);
  iirrpCopy     <<<60, 512, 0, stream>>>(valOffsets,  keyOffsets, commSize);
}
