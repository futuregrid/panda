#ifndef __GPMR_EMITCONFIGURATION_H__
#define __GPMR_EMITCONFIGURATION_H__

#include <panda/VectorOps.h>


namespace panda
{

  class EmitConfiguration
  {
    protected:
      EmitConfiguration(const int totalKeySpaceBytes, const int totalValueSpaceBytes, const int configType);

      enum
      {
        TYPE_ATOMIC = 0,
        TYPE_THREAD,
        TYPE_BLOCK,
        TYPE_GRID,
      };
      int keySpace, valSpace;
      int indexCount, emitsPerThread, keySize, valSize, type;
    public:
      EmitConfiguration();
      EmitConfiguration(const EmitConfiguration & rhs);
      EmitConfiguration & operator = (const EmitConfiguration & rhs);
      static EmitConfiguration createAtomicConfiguration(const int totalKeySpaceBytes, const int totalValueSpaceBytes);
      static EmitConfiguration createThreadConfiguration(const int totalKeySpaceBytes, const int totalValueSpaceBytes,
                                                         const dim3 & gridSize, const dim3 & blockSize);
      static EmitConfiguration createBlockConfiguration (const int totalKeySpaceBytes, const int totalValueSpaceBytes,
                                                         const dim3 & gridSize);
      static EmitConfiguration createGridConfiguration  (const int totalKeySpaceBytes, const int totalValueSpaceBytes,
                                                         const dim3 & gridSize, const dim3 & blockSize, const int totalEmitsPerThread,
                                                         const int pKeySize, const int pValSize);

      inline ~EmitConfiguration() { }

      inline int getKeySpace()        const { return keySpace; }
      inline int getValueSpace()      const { return valSpace; }

      inline bool isAtomic()          const { return type == TYPE_ATOMIC; }
      inline bool isPerThread()       const { return type == TYPE_THREAD; }
      inline bool isPerBlock()        const { return type == TYPE_BLOCK;  }
      inline bool isPerGrid()         const { return type == TYPE_GRID;   }

      inline int getKeySize()         const { return keySize; }
      inline int getValueSize()       const { return valSize; }
      inline int getThreadCount()     const { return indexCount; }
      inline int getEmitsPerThread()  const { return emitsPerThread; }

      inline int getIndexCount()      const { return indexCount; }
  };
}

#endif
