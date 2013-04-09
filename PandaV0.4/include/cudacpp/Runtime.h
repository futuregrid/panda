#ifndef __CUDACPP_RUNTIME_H__
#define __CUDACPP_RUNTIME_H__

#include <oscpp/Timer.h>
#include <cudacpp/Stream.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>


namespace cudacpp
{
  class Array;
  class ChannelFormatDescriptor;
  class DeviceProperties;
  class Error;
  class Event;
  class Stream;
  class String;

  /**
   * This class serves as an interface to the basic runtime operations such as
   * memory allocation and memory management. All functions for this class are
   * static, thus it is not necessary (or even permitted) to instantiate this
   * class.
   *
   * The Runtime class also defines mechanisms for tracking all cuda calls, as
   * well as error reporting and even asserting on CUDA errors.
   *
   * In order to use the Runtime, one must first call Runtime::init(). After
   * that, all calls that one would typically make to CUDA, one should instead
   * make to the equivalent Runtime function.
   *
   * Not all of the CUDA runtime API has been ported. Certain functions like
   * cudaGetSymbolAddress can't be ported for obvious reasons. Other functions
   * are simply "in progress".
   *
   * The vast majority of functions present have already been well described
   * in the CUDA reference manual, please go there for descriptions of their
   * behavior.
   */
  
  class Runtime
  {
    protected:
      /// Runtime flag telling if the Runtime should assert on an error.
      static bool quitOnError;
      /// Runtime flag telling if the Runtime should print errors to stderr.
      static bool errorsToStderr;
      /// Runtime flag telling if the Runtime should log all CUDA calls.
      static bool logCalls;
      /// An STL vector containing all of the cuda calls made.
      static void * cudaCallLog;
      /// A timer for the current program execution.
      static oscpp::Timer timer;
      /// A map of all the current host allocations.
      static void * hostAllocs;
      /// A map of all the current device allocations.
      static void * deviceAllocs;

      /// Default constructor. Protected so users can't call it.
      Runtime();
      /// Default destructor. Protected so users can't call it.
      ~Runtime();
    public:
      /// Initiaize the wrapper library.
      static void         init                          ();
      static void         finalize                      ();
      static int          getLastError                  ();
      static String       getErrorString                (const int errorID);
      static int          getDeviceCount                ();
      static void         setDevice                     (const int deviceID);
      static int          getDevice                     ();
      static int          chooseDevice                  (const DeviceProperties * const desiredProps);
      static void *       mallocDevice                  (const size_t size);
      static void *       mallocPitch                   (size_t * const pitch, const size_t widthInBytes, const size_t height);
      static void *       mallocHost                    (const size_t size);
      static Array *      mallocArray                   (const ChannelFormatDescriptor & channel, const size_t width, const size_t height);
      static void         free                          (void * const ptr);
      static void         freeArray                     (Array * const arr);
      static void         freeHost                      (void * const ptr);
      static void         memcpyHtoD                    (void * const dst, const void * const src, const size_t size);
      static void         memcpyDtoH                    (void * const dst, const void * const src, const size_t size);
      static void         memcpyDtoD                    (void * const dst, const void * const src, const size_t size);
      static void         memcpyToSymbolHtoD            (const char * const dst, const void * const src, const size_t size, const size_t offset);
      static void         memcpyToSymbolDtoH            (const char * const dst, const void * const src, const size_t size, const size_t offset);
      static void         memcpyToSymbolDtoD            (const char * const dst, const void * const src, const size_t size, const size_t offset);
      static void         memcpyHtoDAsync               (void * const dst, const void * const src, const size_t size, Stream * stream);
      static void         memcpyDtoHAsync               (void * const dst, const void * const src, const size_t size, Stream * stream);
      static void         memcpyDtoDAsync               (void * const dst, const void * const src, const size_t size, Stream * stream);
      static void         memcpyToSymbolHtoDAsync       (const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream);
      static void         memcpyToSymbolDtoHAsync       (const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream);
      static void         memcpyToSymbolDtoDAsync       (const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream);
      static void         memset                        (void * const devPtr, const int value, const size_t count);
      static void         memset2D                      (void * const devPtr, const size_t pitch, const int value, const size_t width, const size_t height);
      static void         memcpy2DHtoH                  (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height);
      static void         memcpy2DHtoD                  (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height);
      static void         memcpy2DDtoD                  (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height);
      static void         memcpy2DDtoH                  (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height);
      static void         memcpy2DHtoHAsync             (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream);
      static void         memcpy2DHtoDAsync             (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream);
      static void         memcpy2DDtoDAsync             (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream);
      static void         memcpy2DDtoHAsync             (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream);
      static void         memcpyToArrayHtoH             (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count);
      static void         memcpyToArrayHtoD             (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count);
      static void         memcpyToArrayDtoD             (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count);
      static void         memcpyToArrayDtoH             (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count);
      static void         memcpyToArrayHtoHAsync        (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream);
      static void         memcpyToArrayHtoDAsync        (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream);
      static void         memcpyToArrayDtoDAsync        (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream);
      static void         memcpyToArrayDtoHAsync        (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream);
      static void         memcpyFromArrayHtoH           (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count);
      static void         memcpyFromArrayHtoD           (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count);
      static void         memcpyFromArrayDtoD           (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count);
      static void         memcpyFromArrayDtoH           (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count);
      static void         memcpyFromArrayHtoHAsync      (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream);
      static void         memcpyFromArrayHtoDAsync      (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream);
      static void         memcpyFromArrayDtoDAsync      (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream);
      static void         memcpyFromArrayDtoHAsync      (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream);
      /// Calls cudaThreadSynchronize.
      static void         sync                          ();
      /// Calls cudaThreadExit.
      static void         exit                          ();
      /// @param quit Whether to assert upon a signaled CUDA error.
      static void         setQuitOnError                (const bool quit);
      /// @param print Whether to print CUDA errors to stderr immediately.
      static void         setPrintErrors                (const bool print);
      /// @param print Whether to log all CUDA calls.
      static void         setLogCudaCalls               (const bool log);

      /// Returns the current stack.
      static String getStackTrace();
      /**
       * Print all allocated (device and host) blocks to stdout if the number of remaining
       * host allocations is greater than or equal to hostThresh or the number of remaining device
       * allocations is greater than or equal to deviceThresh.
       */
      static void         printAllocs                   (const unsigned int hostThresh, const unsigned int deviceThresh);
      /**
       * Checks the passed-in error to ensure that the CUDA function behaved
       * appropriately and did not raise any errors.
       *
       * @param error The value to be checked.
       */
      static void         checkCudaError                (const Error & error);
      /**
       * Logs the call, if logging is enabled.
       *
       * @param fmt The printf-style format of the CUDA-call description.
       */
      static void         logCudaCall                   (const char * const fmt, ...);
      /// @param fp The FILE* to which to print the log generated thus far.
      static void         printCudaLog                  (FILE * fp);
  };
}

#endif
