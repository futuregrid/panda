#ifndef __GPMRCPUCONFIG_H__
#define __GPMRCPUCONFIG_H__

typedef struct _GPMRCPUConfig
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
} GPMRCPUConfig;

#endif
