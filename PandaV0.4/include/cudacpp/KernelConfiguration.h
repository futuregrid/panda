#ifndef __PRM_KERNELCONFIGURATION_H__
#define __PRM_KERNELCONFIGURATION_H__

#include <cudacpp/Vector2.h>
#include <cudacpp/Vector3.h>

namespace cudacpp
{
  class Stream;
  class String;

  /**
   * This class holds the four configuration variables for a CUDA kernel launch.
   * These variables are the grid size, block size, amount of shared memory per
   * block, and the stream to use for a block.
   */
  class KernelConfiguration
  {
    protected:
      /// The number and dimensions of blocks to use for the kernel.
      Vector2<int> gridSize;
      /// The number and dimensions of threads to use for the kernel.
      Vector3<int> blockSize;
      /// The amount of shared memory to use for the kernel.
      int shmem;
      /// The stream to use for the kernel.
      Stream * stream;
    public:
      /**
       * Creates a kernel configuration with one dimension of blocks, one
       * dimension of threads, and by default no extra shared memory and no
       * stream.
       *
       * @param numBlocks The number of blocks to use.
       * @param numThreads The number of threads to use.
       * @param sharedMem The amount of extra shared memory to use.
       * @param cudaStream The stream in which to run this kernel.
       */
      KernelConfiguration(int numBlocks, int numThreads, int sharedMem = 0, Stream * const cudaStream = NULL);

      /**
       * Convenience function for creating a multi-dimensional configuration.
       *
       * @param numBlocksX The size of the grid in the first dimension.
       * @param numBlocksY The size of the grid in the second dimension.
       * @param numThreadsX The number of threads in each block in the first
       *                    dimension.
       * @param numThreadsX The number of threads in each block in the second
       *                    dimension.
       * @param numThreadsX The number of threads in each block in the third
       *                    dimension.
       * @param sharedMem The amount of extra shared memory to use.
       * @param cudaStream The stream in which to run this kernel.
       */
      KernelConfiguration(int numBlocksX, int numBlocksY, int numThreadsX, int numThreadsY, int numThreadsZ, int sharedMem = 0, Stream * const cudaStream = NULL);

      /**
       * Creates a kernel configuration with the specified dimensions of blocks
       * and threads, and by default no extra shared memory and no stream.
       *
       * @param pGridSize The size of the grid.
       * @param pBlockSize The size of each block.
       * @param sharedMem The amount of extra shared memory to use.
       * @param cudaStream The stream in which to run this kernel.
       */
      KernelConfiguration(const Vector2<int> & pGridSize, const Vector3<int> & pBlockSize, int sharedMem = 0, Stream * const cudaStream = NULL);

      /**
       * Copy constructor.
       *
       * @param rhs The instance from which to copy parameters.
       */
      KernelConfiguration(const KernelConfiguration & rhs);

      /**
       * Assignment operator.
       *
       * @param rhs The instance from which to copy parameters.
       * @return An updated reference to *this.
       */
      KernelConfiguration & operator = (const KernelConfiguration & rhs);

      /// @return The preferred size of the grid.
      inline const Vector2<int> & getGridSize() const { return gridSize; }


      /// @return The preferred size of each block.
      inline const Vector3<int> & getBlockSize() const { return blockSize; }

      /// @return The preferred amount of extra shared memory per block.
      inline int                  getSharedMemoryUsage() const { return shmem; }

      /// @return The preferred stream in which to run.
      inline Stream *             getStream() const { return stream; }

      /// @return A human-readable string representation of this configuration.
      String toString() const;
  };
}

#endif
