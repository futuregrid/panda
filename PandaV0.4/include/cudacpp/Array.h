#ifndef __CUDACPP_ARRAY_H__
#define __CUDACPP_ARRAY_H__

#include <driver_types.h>

namespace cudacpp
{
  /// THis is a basic wrapper class for CUDA arrays.
  class Array
  {
    protected:
      /// A pointer to a struct cudaArray.
      cudaArray * handle;
    public:
      /**
       * Simple assigns handle from cudaArrayHandle.
       *
       * @param cudaArrayHandle This should be a pointer to a struct cudaArray.
       */
      Array(cudaArray * const cudaArrayHandle);

      /// @return A pointer to handle.
      cudaArray * getHandle();
      /// @return A pointer to handle.
      const cudaArray * getHandle() const;
  };
}

#endif
