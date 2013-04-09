#include <panda/PreLoadedPandaChunk.h>
#include <cudacpp/Runtime.h>

namespace panda
{

	
  int PreLoadedPandaChunk::key = 0;

  PreLoadedPandaChunk::PreLoadedPandaChunk(void * const pData,
                                                   const int pElemSize,
                                                   const int pNumElems)
  {
    data = pData;
    elemSize = pElemSize;
    numElems = pNumElems;
  }
  PreLoadedPandaChunk::~PreLoadedPandaChunk()
  {
  }

  
  bool PreLoadedPandaChunk::updateQueuePosition(const int newPosition)
  {
    return false;
  }
  int PreLoadedPandaChunk::getMemoryRequirementsOnGPU() const
  {
    return elemSize * numElems;
  }
  void PreLoadedPandaChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
  {
    cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, numElems * elemSize, memcpyStream);
  }
  void PreLoadedPandaChunk::finalizeAsync()
  {
  }

  void PreLoadedPandaChunk::finishLoading()
  {

  }

}//PreLoadedPandaChunk

namespace panda
{
/*
	VariousSizePandaChunk::VariousSizePandaChunk(void * const pData, const int dataSize)
	{
		this->data = pData;
		this->dataSize = dataSize;
	}

	VariousSizePandaChunk::~VariousSizePandaChunk()
	{
	}

	int VariousSizePandaChunk::getMemoryRequirementsOnGPU() const
	{
		return dataSize;
	}

	void VariousSizePandaChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
	{
    cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, dataSize, memcpyStream);
	}
*/
}
