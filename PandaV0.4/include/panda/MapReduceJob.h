#ifndef __PANDA_SUPMAPREDUCEJOB_H__
#define __PANDA_SUPMAPREDUCEJOB_H__

#include <vector>

namespace panda
{

  class Message;

  class Chunk;
  class Combiner;
  class Mapper;
  class Partitioner;
  class PartialReducer;
  class Sorter;
  class Reducer;
  class SerializedItemCollection;

  class MapReduceJob
  {
    protected:
      Message         * messager;
      Combiner        * combiner;
      Mapper          * mapper;
      Partitioner     * partitioner;
      PartialReducer  * partialReducer;
      Sorter          * sorter;
      Reducer         * reducer;
      int commRank, commSize, deviceNum;

      void setDevice();
      virtual void map() = 0;
      virtual void sort() = 0;
      virtual void reduce() = 0;
      void collectTimings();
    public:
      MapReduceJob(int & argc, char **& argv);
      virtual ~MapReduceJob();

      inline Message          * getMessage()        { return messager;        }
      inline Combiner        * getCombiner()        { return combiner;        }
      inline Mapper          * getMapper()          { return mapper;          }
      inline Partitioner     * getPartitioner()     { return partitioner;     }
      inline PartialReducer  * getPartialReducer()  { return partialReducer;  }
      inline Sorter          * setSorter()          { return sorter;          }
      inline Reducer         * getReducer()         { return reducer;         }
      inline int               getDeviceNumber()    { return deviceNum;       }

      inline void setMessage       (Message       * const pMessage)         { messager       = pMessage;        }
      inline void setCombiner      (Combiner       * const pCombiner)       { combiner       = pCombiner;       }
      inline void setMapper        (Mapper         * const pMapper)         { mapper         = pMapper;         }
      inline void setPartitioner   (Partitioner    * const pPartitioner)    { partitioner    = pPartitioner;    }
      inline void setPartialReducer(PartialReducer * const pPartialReducer) { partialReducer = pPartialReducer; }
      inline void setSorter        (Sorter         * const pSorter)         { sorter         = pSorter;         }
      inline void setReducer       (Reducer        * const pReducer)        { reducer        = pReducer;        }

      virtual void addInput(Chunk * chunk) = 0;
	  virtual void addMapTasks(Chunk *chunk) = 0;
      virtual void execute() = 0;
  };
}

#endif