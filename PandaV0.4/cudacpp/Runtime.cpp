#include <cudacpp/Runtime.h>
#include <cudacpp/myString.h>
#include <string.h>

namespace cudacpp
{
	void Runtime::finalize(){
	
	}//Runable

	int Runtime::getLastError(){
		return 0;
	}//

	void         Runtime::init(){
		
	}//void

	String Runtime::getErrorString(const int errorID){
		return 0;
	}//String

	int Runtime::getDeviceCount(){

		int gpu_count = 0;
		cudaGetDeviceCount(&gpu_count);
		return gpu_count;

	}//int

	void *  Runtime:: mallocHost (const size_t size){
		return malloc(size);
	}//void

	void Runtime::freeHost(void * const ptr){
		//return 0;
	}//void

	int Runtime::getDevice(){
		return 0;
	}//int

	int Runtime::chooseDevice(const DeviceProperties * const desiredProps){
		return 0;
	}//int

	void * Runtime::mallocDevice(const size_t size){
		int *deviceBuff;
		cudaMalloc((void**)&deviceBuff, size);
		return deviceBuff;
	}//void

    void * Runtime::mallocPitch (size_t * const pitch, const size_t widthInBytes, const size_t height)
	{
		return 0;
	}
	
	Array * Runtime::mallocArray(const ChannelFormatDescriptor & channel, const size_t width, const size_t height){
		return 0;
	}//Array;

	void Runtime::free (void * const ptr){
	
	}//void

	void Runtime::setDevice(const int deviceID)
	{
	}

	
	void      Runtime::freeArray                     (Array * const arr){};
	
	void      Runtime::memcpyHtoD                    (void * const dst, const void * const src, const size_t size)
	{
		cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice );
	};
    void      Runtime::memcpyDtoH                    (void * const dst, const void * const src, const size_t size)
	{
		cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost );
	};

    void      Runtime::memcpyDtoD                    (void * const dst, const void * const src, const size_t size)
	{
		memcpy(dst,src,size);
	};

    void      Runtime::memcpyToSymbolHtoD            (const char * const dst, const void * const src, const size_t size, const size_t offset){};
    void      Runtime::memcpyToSymbolDtoH            (const char * const dst, const void * const src, const size_t size, const size_t offset){};
    void      Runtime::memcpyToSymbolDtoD            (const char * const dst, const void * const src, const size_t size, const size_t offset){};
    void      Runtime::memcpyHtoDAsync               (void * const dst, const void * const src, const size_t size, Stream * stream)
	{
		cudaMemcpyAsync( dst, src, size,cudaMemcpyHostToDevice , stream->getHandle() );
	};

    void      Runtime::memcpyDtoHAsync               (void * const dst, const void * const src, const size_t size, Stream * stream)
	{
		cudaMemcpyAsync( dst, src, size,cudaMemcpyDeviceToHost , stream->getHandle() );
	};

    void      Runtime::memcpyDtoDAsync               (void * const dst, const void * const src, const size_t size, Stream * stream)
	{
		cudaMemcpyAsync( dst, src, size,cudaMemcpyDeviceToDevice , stream->getHandle() );
	};

    void      Runtime::memcpyToSymbolHtoDAsync       (const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream){};
    void      Runtime::memcpyToSymbolDtoHAsync       (const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream){};
    void      Runtime::memcpyToSymbolDtoDAsync       (const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream){};

	  void	  Runtime::memset                        (void * const devPtr, const int value, const size_t count)
	  {
		  
	  };
      void    Runtime::memset2D                      (void * const devPtr, const size_t pitch, const int value, const size_t width, const size_t height){};
      void    Runtime::memcpy2DHtoH                  (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height){};
      void    Runtime::memcpy2DHtoD                  (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height){};
      void    Runtime::memcpy2DDtoD                  (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height){};
      void    Runtime::memcpy2DDtoH                  (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height){};
      void    Runtime::memcpy2DHtoHAsync             (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream){};
      void    Runtime::memcpy2DHtoDAsync             (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream){};
      void    Runtime::memcpy2DDtoDAsync             (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream){};
      void    Runtime::memcpy2DDtoHAsync             (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream){};
      void    Runtime::memcpyToArrayHtoH             (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count){};
      void    Runtime::memcpyToArrayHtoD             (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count){};
      void    Runtime::memcpyToArrayDtoD             (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count){};
      void    Runtime::memcpyToArrayDtoH             (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count){};
      void    Runtime::memcpyToArrayHtoHAsync        (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream){};
      void    Runtime::memcpyToArrayHtoDAsync        (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream){};
      void    Runtime::memcpyToArrayDtoDAsync        (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream){};
      void    Runtime::memcpyToArrayDtoHAsync        (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream){};
      void    Runtime::memcpyFromArrayHtoH           (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count){};
      void    Runtime::memcpyFromArrayHtoD           (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count){};
      void    Runtime::memcpyFromArrayDtoD           (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count){};
      void    Runtime::memcpyFromArrayDtoH           (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count){};
      void    Runtime::memcpyFromArrayHtoHAsync      (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream){};
      void    Runtime::memcpyFromArrayHtoDAsync      (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream){};
      void    Runtime::memcpyFromArrayDtoDAsync      (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream){};
      void    Runtime::memcpyFromArrayDtoHAsync      (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream){};


	  /// Calls cudaThreadSynchronize.
	  void       Runtime::sync ()
	  {
		  cudaThreadSynchronize();
	  };
	  /// Calls cudaThreadExit.
      void       Runtime::exit ()
	  {
		  cudaThreadExit();
	  };

      /// @param quit Whether to assert upon a signaled CUDA error.
	  void       Runtime::setQuitOnError (const bool quit){};
	  /// @param print Whether to print CUDA errors to stderr immediately.
	  void       Runtime::setPrintErrors (const bool print){};
	  /// @param print Whether to log all CUDA calls.

	  void       Runtime::setLogCudaCalls               (const bool log){};

	  /// Returns the current stack.
	  String Runtime::getStackTrace(){ 
		  return 0; 
	  }
       /**
       * Print all allocated (device and host) blocks to stdout if the number of remaining
       * host allocations is greater than or equal to hostThresh or the number of remaining device
       * allocations is greater than or equal to deviceThresh.
       */
	  void   Runtime::printAllocs  (const unsigned int hostThresh, const unsigned int deviceThresh){};
       /**
       * Checks the passed-in error to ensure that the CUDA function behaved
       * appropriately and did not raise any errors.
       *
       * @param error The value to be checked.
       */
	  void   Runtime::checkCudaError  (const Error & error){};
       /**
       * Logs the call, if logging is enabled.
       *
       * @param fmt The printf-style format of the CUDA-call description.
       */
	  void  Runtime::logCudaCall  (const char * const fmt, ...){};
        /// @param fp The FILE* to which to print the log generated thus far.
	  void  Runtime::printCudaLog (FILE * fp){};

}//namespace
     
        
      
      
      
      

