#include <cstdio>

static const int NUM_THREADS = 128;
static const int MAX_BLOCKS  = 32768;

__global__ void pandaIntIntSorterComputeA(const int blocksSoFar, const int * const inputKeys, int * const uniqueFlags, const int numKeys)
{
  const int index = (blocksSoFar + blockIdx.x) * blockDim.x + threadIdx.x;
  if      (index == 0)                                uniqueFlags[index] = 1;
  else if (inputKeys[index] != inputKeys[index - 1])  uniqueFlags[index] = 1;
  else                                                uniqueFlags[index] = 0;
}
__global__ void gpmrIntIntSorterComputeC(const int * const gpuA, const int * const gpuB, int * const gpuC, const int numKeys)
{
  __shared__ int b0;
  if (threadIdx.x == 0) b0 = gpuB[0];
  __syncthreads();
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numKeys; index += gridDim.x * blockDim.x)
  {
    if (gpuA[index]) gpuC[b0 - gpuB[index]] = index;
  }
}
__global__ void gpmrIntIntSorterComputeD(const int blocksSoFar, const int * const gpuC, int * const gpuD, const int numUniqueKeys, const int numKeys)
{
  const int index = (blocksSoFar + blockIdx.x) * blockDim.x + threadIdx.x;
  if      (index == numUniqueKeys - 1)  gpuD[index] = numKeys - gpuC[index];
  else                                  gpuD[index] = gpuC[index + 1] - gpuC[index];
}
__global__ void gpmrIntIntSorterSetCompactedKeysKernel(const int blocksSoFar, const int * const keys, const int * const input, int * const output, const int numKeys)
{
  const int index = (blocksSoFar + blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < numKeys) output[index] = keys[input[index]];
}
void gpmrIntIntSorterMarkUnique(const void * const gpuInputKeys, void * const gpuUniqueFlags, const int numKeys)
{
  const int NUM_BLOCKS = (numKeys + NUM_THREADS - 1) / NUM_THREADS;
  int blocksSoFar = 0;
  int numBlocks;
  while (blocksSoFar < NUM_BLOCKS)
  {
    numBlocks = (NUM_BLOCKS - blocksSoFar > MAX_BLOCKS ? MAX_BLOCKS : NUM_BLOCKS - blocksSoFar);
    /*gpmrIntIntSorterComputeA<<<numBlocks, NUM_THREADS>>>(blocksSoFar,
                                                         reinterpret_cast<const int * >(gpuInputKeys),
                                                         reinterpret_cast<int * >(gpuUniqueFlags),
                                                         numKeys);
	*/
    blocksSoFar += numBlocks;
  }
}
void gpmrIntIntSorterFindOffsets(const void * const gpuKeys, const void * const gpuA, const void * const gpuB, void * const gpuC, void * const gpuD, const int numKeys, const int numUniqueKeys)
{
  const int NUM_BLOCKS_1 = 60;
  const int NUM_BLOCKS = (numUniqueKeys + NUM_THREADS - 1) / NUM_THREADS;
  int blocksSoFar = 0;
  int numBlocks;

  gpmrIntIntSorterComputeC<<<NUM_BLOCKS_1, NUM_THREADS>>>(reinterpret_cast<const int * >(gpuA),
                                                          reinterpret_cast<const int * >(gpuB),
                                                          reinterpret_cast<int * >(gpuC),
                                                          numKeys);

  while (blocksSoFar < NUM_BLOCKS)
  {
    numBlocks = (NUM_BLOCKS - blocksSoFar > MAX_BLOCKS ? MAX_BLOCKS : NUM_BLOCKS - blocksSoFar);
    gpmrIntIntSorterComputeD<<<numBlocks, NUM_THREADS>>>(blocksSoFar,
                                                         reinterpret_cast<const int * >(gpuC),
                                                         reinterpret_cast<int * >(gpuD),
                                                         numUniqueKeys,
                                                         numKeys);
    blocksSoFar += numBlocks;
  }
}
void gpmrIntIntSorterSetCompactedKeys(const void * const gpuKeys, const void * const gpuInput, void * const gpuOutput, const int numUniqueKeys)
{
  const int NUM_BLOCKS = (numUniqueKeys + NUM_THREADS - 1) / NUM_THREADS;
  int blocksSoFar = 0;
  int numBlocks;
  while (blocksSoFar < NUM_BLOCKS)
  {
    numBlocks = (NUM_BLOCKS - blocksSoFar > MAX_BLOCKS ? MAX_BLOCKS : NUM_BLOCKS - blocksSoFar);
    gpmrIntIntSorterSetCompactedKeysKernel<<<numBlocks, NUM_THREADS>>>(blocksSoFar,
                                                                       reinterpret_cast<const int * >(gpuKeys),
                                                                       reinterpret_cast<const int * >(gpuInput),
                                                                       reinterpret_cast<      int * >(gpuOutput),
                                                                       numUniqueKeys);
    blocksSoFar += numBlocks;
  }
}
