#ifndef __KMEANSMAPPER_H__
#define __KMEANSMAPPER_H__

#include <panda/Mapper.h>

class WCMapper : public panda::Mapper
{
  protected:
    //static const int NUM_BLOCKS = 60;
    int dataSize;
  public:
    WCMapper();
    virtual ~WCMapper();

    virtual panda::EmitConfiguration getEmitConfiguration(panda::Chunk * const chunk) const;
    virtual bool canExecuteOnGPU() const;
    virtual bool canExecuteOnCPU() const;
    virtual void init();
    virtual void finalize();
    virtual void executeOnGPUAsync(panda::Chunk * const chunk, PandaGPUConfig & pandaGPUConfig, void * const gpuMemoryForChunk,
                                   cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream);
    virtual void executeOnCPUAsync(panda::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig);

    //void setCenters(const float * const pCenters);
    //inline float * getGPUCenters() { return gpuCenters; }
    //inline int getNumCenters() const { return numCenters; }
    //inline int getNumDims() const { return numDims; }
};

#endif
