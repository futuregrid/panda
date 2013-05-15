#ifndef __GPMRGPUCONFIG_H__
#define __GPMRGPUCONFIG_H__

typedef struct _PandaGPUConfig
{
  void * keySpace;
  void * valueSpace;
  union
  {
    struct
    {
      int keyOffset;
      int valueOffset;
    } atomic;
    struct
    {
      int * totalKeySize;
      int * totalValueSize;
      int * keyOffset;
      int * valueOffset;
    } thread, block;
    struct
    {
      int numThreads;
      int emitsPerThread;
    } grid;
  } emitInfo;
} PandaGPUConfig;


  


#endif
