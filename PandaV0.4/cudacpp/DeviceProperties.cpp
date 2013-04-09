#include <cudacpp/DeviceProperties.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cudacpp
{

	   /// Default constructor.
	DeviceProperties::DeviceProperties(){}
	DeviceProperties::~DeviceProperties(){}

	DeviceProperties * DeviceProperties::create( const String & pName,
                                        const int maxThreadsPerBlock,
                                        const Vector3<int> & pMaxThreadsDim,
                                        const Vector3<int> & pMaxGridSize,
                                        const int pSharedMemPerBlock,
                                        const int pWarpSize,
                                        const int MemPitch,
                                        const int pRegsPerBlock,
                                        const int pClockRate,
                                        const int pMajor,
                                        const int pMinor,
                                        const int pMultiProcessorCount,
                                        const size_t pTotalConstantMemory,
                                        const size_t pTotalMemBytes,
										const size_t pTextureAlign)
	{
		return 0;
	}

	DeviceProperties * DeviceProperties::get(const int deviceID)
	{
	int gpu_count = 0;
	cudaGetDeviceCount(&gpu_count);
	if (deviceID>=gpu_count){
		printf("wrong deviceID:%d\n",deviceID);
		//TODO exit(-1);
	}
	cudaDeviceProp *gpu_dev = new cudaDeviceProp;
	cudaGetDeviceProperties(gpu_dev, deviceID);
	return 0;
	}//DeviceProperties

	const String & DeviceProperties::getName() const{
		return 0;
	}

	int DeviceProperties::getMaxThreadsPerBlock() const{
		return 0;
	}//int

	const Vector3<int> & DeviceProperties::getMaxBlockSize() const{
		Vector3<int> result;
		return result;
	}

      
	const Vector3<int> & DeviceProperties::getMaxGridSize() const{
		Vector3<int> result;
		return result;
	}
      /// @return The total amount of shared memory available to a SM.
	int           DeviceProperties::getSharedMemoryPerBlock() const{
		return 0;
	}
      /// @return The total amount of constant memory available to a kernel.
	size_t        DeviceProperties::getTotalConstantMemory() const{
		return 0;
	}
      /// @return The SIMD size of each SM.
	int           DeviceProperties::getWarpSize() const{
		return 0;
	}

      /// @return The pitch of global device memory.
	int           DeviceProperties::getMemoryPitch() const{
		return 0;
	}
      /// @return The total number of 32-bit registers per SM.
	int           DeviceProperties::getRegistersPerBlock() const{
		return 0;
	}
      /// @return The frequency of the clock, in kHz.
	int           DeviceProperties::getClockRate() const{
		return 0;
	}
      /// @return The alignment requirements for textures in memory.
	size_t        DeviceProperties::getTextureAlignment() const{
		return 0;
	}
      /// @return The total amount of global device memory.
	size_t        DeviceProperties::getTotalMemory() const{
		return 0;
	}
      /// @return The device's compute capability major number.
	int           DeviceProperties::getMajor() const{
		return 0;
	}
      /// @return The device's compute capability minor number.
	int           DeviceProperties::getMinor() const{
		return 0;
	}
      /// @return The number of SMs on the device.
	int           DeviceProperties::getMultiProcessorCount() const{
		return 0;
	}

}