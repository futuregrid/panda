#include <panda/EmitConfiguration.h>
//#include <cstring>
//#include <stdlib.h>
//#include <string.h>

//#include <cstdlib>
#include <cstring>


namespace panda
{

  EmitConfiguration::EmitConfiguration(const int totalKeySpaceBytes, const int totalValueSpaceBytes, const int configType)
    : keySpace(totalKeySpaceBytes), valSpace(totalValueSpaceBytes), type(configType)
  {
  }

  EmitConfiguration::EmitConfiguration()
  {
    keySpace = valSpace = -1;
    indexCount = emitsPerThread = keySize = valSize = type = -1;
  }

  EmitConfiguration::EmitConfiguration(const EmitConfiguration & rhs)
  {
    memcpy(this, &rhs, sizeof(rhs));
  }

  EmitConfiguration & EmitConfiguration::operator = (const EmitConfiguration & rhs)
  {
    memcpy(this, &rhs, sizeof(rhs));
    return * this;
  }

  EmitConfiguration EmitConfiguration::createAtomicConfiguration(const int totalKeySpaceBytes, const int totalValueSpaceBytes)
  {
    EmitConfiguration ret(totalKeySpaceBytes, totalValueSpaceBytes, TYPE_ATOMIC);
    ret.indexCount = 1;
    ret.keySize = -1;
    ret.valSize = -1;
    return ret;
  }

  EmitConfiguration EmitConfiguration::createThreadConfiguration(const int totalKeySpaceBytes, const int totalValueSpaceBytes,
                                                                 const dim3 & gridSize, const dim3 & blockSize)
  {
    EmitConfiguration ret(totalKeySpaceBytes, totalValueSpaceBytes, TYPE_THREAD);
    ret.indexCount = gridSize.x * gridSize.y * blockSize.x * blockSize.y * blockSize.z;
    ret.keySize = -1;
    ret.valSize = -1;
    return ret;
  }

  EmitConfiguration EmitConfiguration::createBlockConfiguration (const int totalKeySpaceBytes, const int totalValueSpaceBytes,
                                                                 const dim3 & gridSize)
  {
    EmitConfiguration ret(totalKeySpaceBytes, totalValueSpaceBytes, TYPE_BLOCK);
    ret.indexCount = gridSize.x * gridSize.y;
    ret.keySize = -1;
    ret.valSize = -1;
    return ret;
  }
  EmitConfiguration EmitConfiguration::createGridConfiguration  (const int totalKeySpaceBytes, const int totalValueSpaceBytes,
                                                                 const dim3 & gridSize, const dim3 & blockSize, const int totalEmitsPerThread,
                                                                 const int pKeySize, const int pValSize)
  {
    EmitConfiguration ret(totalKeySpaceBytes, totalValueSpaceBytes, TYPE_GRID);
    ret.indexCount = gridSize.x * gridSize.y * blockSize.x * blockSize.y * blockSize.z * totalEmitsPerThread;
    ret.emitsPerThread = totalEmitsPerThread;
    ret.keySize = pKeySize;
    ret.valSize = pValSize;
    ret.keySpace = pKeySize * ret.indexCount;
    ret.valSpace = pValSize * ret.indexCount;
    return ret;
  }

}
