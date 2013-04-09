#ifndef __panda_Message_H__
#define __panda_Message_H__

#include <oscpp/AsyncIORequest.h>
#include <oscpp/Runnable.h>

namespace panda
{
  class Message : public oscpp::Runnable
  {
    protected:
      int commSize, commRank;
    public:
      Message();
      virtual ~Message();

      virtual oscpp::AsyncIORequest * sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
                                             const int keySize,
                                             const int valSize) = 0;
      virtual oscpp::AsyncIORequest * sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
                                             int * const keySizes,
                                             int * const valSizes,
                                             const int numKeys,
                                             const int numVals) = 0;
      virtual void init();
      virtual void finalize() = 0;
      virtual void run() = 0;
      virtual oscpp::AsyncIORequest * finish() = 0;
      virtual void getFinalDataSize(int & keySize, int & valSize) const = 0;
      virtual void getFinalDataSize(int & keySize, int & valSize, int & numKeys, int & numVals) const = 0;
      virtual void getFinalData(void * keyStorage, void * valStorage) const = 0;
      virtual void getFinalData(void * keyStorage, void * valStorage, int * keySizes, int * valSizes) const = 0;
  };
}

#endif
