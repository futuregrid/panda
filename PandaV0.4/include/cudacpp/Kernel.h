#ifndef __CUDACPP_KERNEL_H__
#define __CUDACPP_KERNEL_H__

#include <string.h>

namespace cudacpp
{
  class KernelConfiguration;
  class KernelParameters;
  class String;

  /**
   * The Kernel interface presents a mechanism to wrap a kernel and its
   * invocation. Also contained is a method used to describe a kernel
   * invocation (useful for logging).
   */
  class Kernel
  {
    protected:
      /**
       * Function to be implemented for each kernel instance. This function
       * should either directly invoke the kernel or call a chain of functions
       * which will invoke the kernel.
       *
       * @param config CUDA configuration variables.
       * @param params Kernel-specific parameters.
       */
      virtual void runSub(const KernelConfiguration & config, const KernelParameters & params) = 0;

      /**
       * This function 'may' be overridden by a user to better describe their
       * kernel and the parameters of the kernel. Given the input, the function
       * should decode the index<sup>th</sup> parameter from params and return
       * a useful and human-readable string representation thereof.
       */
      virtual String paramToString(const int index, const KernelParameters & params) const;
    public:
      /// Default constructor.
      Kernel();
      /// Default virtual destructor.
      virtual ~Kernel();

      /**
       * Call this function to invoke a kernel on the current device.
       *
       * @param config CUDA configuration variables.
       * @param params Kernel-specific parameters.
       */
      void run(const KernelConfiguration & config, const KernelParameters & params);

      /**
       * @return The name of the kernel in a string. Useful for debugging and
       *         logging.
       */
      virtual String name() const = 0;
  };
}

#endif
