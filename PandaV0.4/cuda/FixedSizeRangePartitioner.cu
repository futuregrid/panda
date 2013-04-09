#include <panda/PandaGPUConfig.h>
#include <sm_11_atomic_functions.h>

#include <cstdio>

const int NUM_THREADS_IN_PARTITION  = 128;
const int WARP_SIZE                 =  32;
const int LOG_CTA_SIZE              =   7;
const int LOG_WARP_SIZE             =   5;

template <int MAX_LEVEL>
__device__ int warpscan(const int val, volatile int * array)
{
  int index = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE - 1));
  array[index] = 0;
  index += WARP_SIZE;
  array[index] = val;

  if (0 <= MAX_LEVEL) array[index] += array[index -  1];
  if (1 <= MAX_LEVEL) array[index] += array[index -  2];
  if (2 <= MAX_LEVEL) array[index] += array[index -  4];
  if (3 <= MAX_LEVEL) array[index] += array[index -  8];
  if (4 <= MAX_LEVEL) array[index] += array[index - 16];

  return array[index - 1];      // convert inclusive -> exclusive
}
__device__ void scanWarps(const int x, const int y, volatile int * array)
{
  int val0 = warpscan<4>(x, array);
  int val1 = warpscan<4>(y, array);

  int index = threadIdx.x;
  if ((index & 31) == 31)
  {
    array[index >> 5]                = val0 + x;
    array[(index + blockDim.x) >> 5] = val1 + y;
  }
  __syncthreads();
  array[index] = warpscan<LOG_CTA_SIZE - LOG_WARP_SIZE + 1>(array[index], array);
  __syncthreads();

  val0 += array[index >> 5];
  val1 += array[(index + blockDim.x) >> 5];

  __syncthreads();

  array[index]              = val0;
  array[index + blockDim.x] = val1;
}
__device__ void scan(volatile int * array)
{
  int val0 = array[threadIdx.x];
  int val1 = array[threadIdx.x + blockDim.x];
  scanWarps(val0, val1, array);
  __syncthreads();
}
__device__ int reduce(int * array)
{
  int size = blockDim.x / 2;
  __syncthreads();
  while (size > 0)
  {
    if (threadIdx.x < size) array[threadIdx.x] += array[threadIdx.x + size];
    size /= 2;
    __syncthreads();
  }
  return array[0];
}
__device__ int countNonZero(int * array)
{
  __shared__ int temp[NUM_THREADS_IN_PARTITION];
  temp[threadIdx.x] = (array[threadIdx.x] == 0 ? 0 : 1);
  return reduce(temp);
}
__device__ int compact(int * const array,
                       int * const aux0,
                       int * const aux1)
{
  __shared__ int leftShift[NUM_THREADS_IN_PARTITION * 2];
  leftShift[threadIdx.x]              = (array[threadIdx.x]               == 0 ? 1 : 0);
  leftShift[threadIdx.x + blockDim.x] = (array[threadIdx.x + blockDim.x]  == 0 ? 1 : 0);
  __syncthreads();
  scan(leftShift);
  const int tempVal0 = array[threadIdx.x];
  const int tempVal1 = array[threadIdx.x  + blockDim.x];
  const int temp00 = aux0[threadIdx.x];
  const int temp01 = aux0[threadIdx.x + blockDim.x];
  const int temp10 = aux1[threadIdx.x];
  const int temp11 = aux1[threadIdx.x + blockDim.x];
  __syncthreads();
  if (tempVal0 != 0)
  {
    array[threadIdx.x - leftShift[threadIdx.x]] = tempVal0;
    aux0 [threadIdx.x - leftShift[threadIdx.x]] = temp00;
    aux1 [threadIdx.x - leftShift[threadIdx.x]] = temp10;
  }
  if (tempVal1 != 0)
  {
    array[threadIdx.x + blockDim.x - leftShift[threadIdx.x + blockDim.x]] = tempVal1;
    aux0 [threadIdx.x + blockDim.x - leftShift[threadIdx.x + blockDim.x]] = temp01;
    aux1 [threadIdx.x + blockDim.x - leftShift[threadIdx.x + blockDim.x]] = temp11;
  }
  __syncthreads();
  return blockDim.x * 2 - leftShift[blockDim.x * 2 - 1];
} 

__global__ void PandaRangePartitionerZeroCountsAndIndicesKernel(int * gpuBucketCounts, int * gpuBucketIndices)
{
  if (blockIdx.x == 0)  gpuBucketCounts [threadIdx.x] = 0;
  else                  gpuBucketIndices[threadIdx.x] = 0;
}
void PandaRangePartitionerZeroCountsAndIndices(const int commSize, int * gpuBucketCounts, int * gpuBucketIndices, cudaStream_t * stream)
{
  dim3 blockSize(commSize, 1, 1);
  dim3 gridSize(2, 1, 1);
  PandaRangePartitionerZeroCountsAndIndicesKernel<<<gridSize, blockSize, 0, *stream>>>(gpuBucketCounts, gpuBucketIndices);
}

__global__ void PandaRangePartitionerCountBucketKernel(const int commSize, const int numKeys, const int bucket, int * gpuKeys, int * gpuBucketCount)
{
  __shared__ int addToCount[512];
  addToCount[threadIdx.x] = 0;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  addToCount[threadIdx.x] = (index < numKeys && ((gpuKeys[blockIdx.x * blockDim.x + threadIdx.x] % commSize) == bucket)) ? 1 : 0;
  reduce(addToCount);
  if (threadIdx.x == 0) atomicAdd(gpuBucketCount, addToCount[0]);
}
void PandaRangePartitionerCount(const int commSize, const int numKeys, int * gpuKeys, int * gpuBucketCounts, cudaStream_t * stream)
{
  dim3 blockSize(512, 1, 1);
  dim3 gridSize((numKeys + 511) / 512, 1, 1);
  for (int i = 0; i < commSize; ++i)
  {
    PandaRangePartitionerCountBucketKernel<<<gridSize, blockSize, 0, *stream>>>(commSize, numKeys, i, gpuKeys, gpuBucketCounts + i);
  }
}

__global__ void PandaRangePartitionerCountKernel(const int commSize, const int bucket, const int numKeys, int * gpuKeys, int * gpuVals, int * gpuTempKeySpace, int * gpuTempValSpace, int * gpuKeyOffsets, int * gpuBucketIndices)
{
  __shared__ int globalOffset, count;
  __shared__ int temp[NUM_THREADS_IN_PARTITION * 2];
  __shared__ int keys[NUM_THREADS_IN_PARTITION * 2];
  __shared__ int vals[NUM_THREADS_IN_PARTITION * 2];

  for (int i = blockIdx.x * blockDim.x; i < numKeys; i += blockDim.x * gridDim.x)
  {
    temp[threadIdx.x] = 0;
    temp[threadIdx.x + blockDim.x] = 0;
    if (i + threadIdx.x < numKeys)
    {
      keys[threadIdx.x] = gpuKeys[i + threadIdx.x];
      vals[threadIdx.x] = gpuVals[i + threadIdx.x];
      temp[threadIdx.x] = keys[threadIdx.x] % commSize == bucket ? 1 : 0;
    }
    if (i + threadIdx.x + blockDim.x < numKeys)
    {
      keys[threadIdx.x + blockDim.x] = gpuKeys[i + threadIdx.x + blockDim.x];
      vals[threadIdx.x + blockDim.x] = gpuVals[i + threadIdx.x + blockDim.x];
      temp[threadIdx.x + blockDim.x] = keys[threadIdx.x + blockDim.x] % commSize == bucket ? 1 : 0;
    }
    count = reduce(temp);
    if (threadIdx.x == 0) globalOffset = atomicAdd(gpuKeyOffsets + bucket, temp[0] * sizeof(int)) / sizeof(int);
    __syncthreads();
    temp[threadIdx.x]               = ((i + threadIdx.x               < numKeys) && (keys[threadIdx.x]              % commSize == bucket)) ? 1 : 0;
    temp[threadIdx.x + blockDim.x]  = ((i + threadIdx.x + blockDim.x  < numKeys) && (keys[threadIdx.x + blockDim.x] % commSize == bucket)) ? 1 : 0;
    compact(temp, keys, vals);
    __syncthreads();
    if (threadIdx.x < count)
    {
      gpuTempKeySpace[globalOffset + threadIdx.x             ] = keys[threadIdx.x             ];
      gpuTempValSpace[globalOffset + threadIdx.x             ] = vals[threadIdx.x             ];
    }
    if (threadIdx.x + blockDim.x < count)
    {
      gpuTempKeySpace[globalOffset + threadIdx.x + blockDim.x] = keys[threadIdx.x + blockDim.x];
      gpuTempValSpace[globalOffset + threadIdx.x + blockDim.x] = vals[threadIdx.x + blockDim.x];
    }
  }
}

void PandaRangePartitionerCount(const int commSize, const int numKeys, int * gpuKeys, int * gpuVals, int * gpuTempKeySpace, int * gpuTempValSpace, int * gpuKeyOffsets, int * gpuBucketIndices, cudaStream_t * stream)
{
  dim3 blockSize(NUM_THREADS_IN_PARTITION, 1, 1);
  dim3 gridSize(1024, 1, 1);
  for (int i = 0; i < commSize; ++i)
  {
    PandaRangePartitionerCountKernel<<<gridSize, blockSize, 0, *stream>>>(commSize, i, numKeys, gpuKeys, gpuVals, gpuTempKeySpace, gpuTempValSpace, gpuKeyOffsets, gpuBucketIndices);
  }
}

__global__ void PandaRangePartitionerMoveFromTempKernel(const int numKeys, int * gpuKeys, int * gpuVals, int * gpuTempKeySpace, int * gpuTempValSpace)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numKeys)
  {
    gpuKeys[index] = gpuTempKeySpace[index];
    gpuVals[index] = gpuTempValSpace[index];
  }
}
void PandaRangePartitionerMoveFromTemp(const int commSize, const int numKeys, int * gpuKeys, int * gpuVals, int * gpuTempKeySpace, int * gpuTempValSpace, cudaStream_t * stream)
{
  dim3 blockSize(512, 1, 1);
  dim3 gridSize((numKeys + 511) / 512, 1, 1);
  PandaRangePartitionerMoveFromTempKernel<<<gridSize, blockSize, 0, *stream>>>(numKeys, gpuKeys, gpuVals, gpuTempKeySpace, gpuTempValSpace);
}

__global__ void PandaRangePartitionerSetKeyAndValCountsKernel(const int * const keyCounts, int * const valCounts)
{
  valCounts[threadIdx.x] = keyCounts[threadIdx.x];
}
void PandaRangePartitionerSetKeyAndValCounts(const int commSize, int * gpuKeyCounts, int * gpuValCounts, cudaStream_t * stream)
{
  dim3 blockSize(commSize, 1, 1);
  dim3 gridSize(1, 1, 1);
  PandaRangePartitionerSetKeyAndValCountsKernel<<<gridSize, blockSize, 0, *stream>>>(gpuKeyCounts, gpuValCounts);
}

__global__ void PandaRangePartitionerSetKeyAndValOffsetsKernel(const int commSize, const int * const gpuKeyCounts, int * const gpuKeyOffsets, int * const gpuValOffsets)
{
  int temp = gpuKeyCounts[0];
  gpuKeyOffsets[0] = 0;
  gpuValOffsets[0] = 0;
  for (int i = 1; i < commSize; ++i)
  {
    gpuKeyOffsets[i] = temp * sizeof(int);
    gpuValOffsets[i] = temp * sizeof(int);
    temp += gpuKeyCounts[i];
  }
}
void PandaRangePartitionerSetKeyAndValOffsets(const int commSize, const int * const gpuKeyCounts, int * const gpuKeyOffsets, int * const gpuValOffsets, cudaStream_t * stream)
{
  dim3 blockSize(1, 1, 1);
  dim3 gridSize(1, 1, 1);
  PandaRangePartitionerSetKeyAndValOffsetsKernel<<<gridSize, blockSize, 0, *stream>>>(commSize, gpuKeyCounts, gpuKeyOffsets, gpuValOffsets);
}
