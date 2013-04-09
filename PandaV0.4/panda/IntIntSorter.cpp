#include <mpi.h>
#include <panda/IntIntSorter.h>
#include <cudacpp/Runtime.h>
#include <cudpp/cudpp.h>
//#include <mpi.h>

#include <algorithm>

void gpmrIntIntSorterMarkUnique(const void * const gpuInputKeys, void * const gpuUniqueFlags, const int numKeys);
void gpmrIntIntSorterFindOffsets(const void * const gpuKeys, const void * const gpuA, const void * const gpuB, void * const gpuC, void * const gpuD, const int numKeys, const int numUniqueKeys);
void gpmrIntIntSorterSetCompactedKeys(const void * const gpuKeys, const void * const gpuInput, void * const gpuOutput, const int numUniqueKeys);

namespace panda
{

  namespace intintsorter
  {
    struct IndexOffsetCount
    {
      int index;
      int offset;
      int count;
      inline IndexOffsetCount() : count(0) { }
      inline IndexOffsetCount(const IndexOffsetCount & rhs) : index(rhs.index), offset(rhs.offset), count(rhs.count) { }
      inline IndexOffsetCount & operator = (const IndexOffsetCount & rhs)
      {
        index = rhs.index;
        offset = rhs.offset;
        count = rhs.count;
        return * this;
      }
      inline void upCount() { ++count; }
    };

    template <typename Key, typename Value>
    void quickSort(Key * const keys, Value * const vals, const int numItems, int depth = 0)
    {
      if (numItems <= 1) return;
      if (numItems == 2)
      {
        if (keys[0] > keys[1])
        {
          std::swap(keys[0], keys[1]);
          std::swap(vals[0], vals[1]);
        }
        return;
      }
      const unsigned int randPivot0 = (static_cast<unsigned int>(rand()) * static_cast<unsigned int>(rand())) % numItems;
      const unsigned int randPivot1 = (static_cast<unsigned int>(rand()) * static_cast<unsigned int>(rand())) % numItems;
      const unsigned int randPivot2 = (static_cast<unsigned int>(rand()) * static_cast<unsigned int>(rand())) % numItems;
      unsigned int randPivot;

      if      (keys[randPivot0] <= keys[randPivot1] && keys[randPivot0] >= keys[randPivot2] || keys[randPivot0] >= keys[randPivot1] && keys[randPivot0] <= keys[randPivot2]) randPivot = randPivot0;
      else if (keys[randPivot1] <= keys[randPivot0] && keys[randPivot1] >= keys[randPivot2] || keys[randPivot1] >= keys[randPivot0] && keys[randPivot1] <= keys[randPivot2]) randPivot = randPivot1;
      else if (keys[randPivot2] <= keys[randPivot1] && keys[randPivot2] >= keys[randPivot0] || keys[randPivot2] >= keys[randPivot1] && keys[randPivot2] <= keys[randPivot0]) randPivot = randPivot2;

      int storeIndex = 0;

      std::swap(keys[randPivot], keys[numItems - 1]);
      std::swap(vals[randPivot], vals[numItems - 1]);

      for (int i = 0; i < numItems - 1; ++i)
      {
        if (keys[i] < keys[numItems - 1])
        {
          std::swap(keys[i], keys[storeIndex]);
          std::swap(vals[i], vals[storeIndex]);
          ++storeIndex;
        }
      }
      std::swap(keys[storeIndex], keys[numItems - 1]);
      std::swap(vals[storeIndex], vals[numItems - 1]);
      quickSort<Key, Value>(keys,                  vals,                  storeIndex,                depth + 1);
      quickSort<Key, Value>(keys + storeIndex + 1, vals + storeIndex + 1, numItems - storeIndex - 1, depth + 1);
    }

  }

  IntIntSorter::IntIntSorter()
  {
  }
  IntIntSorter::~IntIntSorter()
  {
  }

  bool IntIntSorter::canExecuteOnGPU() const
  {
    return true;
  }
  bool IntIntSorter::canExecuteOnCPU() const
  {
    return true;
  }
  void IntIntSorter::init()
  {
  }

  void IntIntSorter::finalize()
  {
  }//void

  void IntIntSorter::executeOnGPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals)
  {
    if (numKeys == 0)
    {
      numUniqueKeys = 0;
      *keyOffsets = *valOffsets = NULL;
      *numVals = NULL;
      return;
    }//if

    if (numKeys > 32 * 1048576)
    {
      executeOnCPUAsync(keys, vals, numKeys, numUniqueKeys, keyOffsets, valOffsets, numVals);
      return;
    }//if

    int commRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    CUDPPConfiguration cudppConfig;
    CUDPPHandle planHandle;

    void * gpuInputKeys   = cudacpp::Runtime::mallocDevice(sizeof(int) * numKeys);
    void * gpuInputVals   = cudacpp::Runtime::mallocDevice(sizeof(int) * numKeys);
    void * gpuUniqueFlags = cudacpp::Runtime::mallocDevice(sizeof(int) * numKeys);
    void * gpuValOffsets  = cudacpp::Runtime::mallocDevice(sizeof(int) * numKeys);

    cudacpp::Runtime::memcpyHtoD(gpuInputKeys, keys, sizeof(int) * numKeys);
    cudacpp::Runtime::memcpyHtoD(gpuInputVals, vals, sizeof(int) * numKeys);
	
    /*
      what we need to get out of here:
	
      1 - sorted keys and values
      2 - num unique keys
      3 - number of values for each key
      4 - value offsets
      5 - compacted keys
	
      to get:
          simply sort
          A = find unique values
          B = reverse exclusive scan of "A"
          C = if A[i] == 1
                C[B[0] - B[i]] = i
          D = [0] = C[0] + 1
              [N] = #keys - C[#keys - 1]
              [i] = C[i + 1] - C[i]
          E = forward exclusive scan D
          F = keys[E[i]]
	
      1 = result of sort (only copy the values)
      2 = B[0]
      3 = D
      4 = E
      5 = F
    */
	
    // 1
	
    cudppConfig.algorithm  = CUDPP_SORT_RADIX;
    cudppConfig.op         = CUDPP_ADD; // ignored
    cudppConfig.datatype   = CUDPP_UINT;
    cudppConfig.options    = CUDPP_OPTION_KEY_VALUE_PAIRS;
    
	cudppPlan(&planHandle, cudppConfig, numKeys, 1, numKeys * sizeof(int));
    cudppSort(planHandle, gpuInputKeys, gpuInputVals, sizeof(int) * 8, numKeys);
		
	cudppDestroyPlan(planHandle);
    cudacpp::Runtime::sync();
	
    cudacpp::Runtime::memcpyDtoH(keys, gpuInputKeys, sizeof(int) * numKeys);
    cudacpp::Runtime::memcpyDtoH(vals, gpuInputVals, sizeof(int) * numKeys);
	
    // 2 - A = gpuUniqueFlags
    gpmrIntIntSorterMarkUnique(gpuInputKeys, gpuUniqueFlags, numKeys);
	
    // 2 - B = gpuValOffsets
    cudppConfig.algorithm  = CUDPP_SCAN;
    cudppConfig.op         = CUDPP_ADD; // ignored
    cudppConfig.datatype   = CUDPP_INT;
    cudppConfig.options    = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_BACKWARD;
    cudppPlan(&planHandle, cudppConfig, numKeys, 1, numKeys * sizeof(int));
    cudppScan(planHandle, gpuValOffsets, gpuUniqueFlags, numKeys);
    cudppDestroyPlan(planHandle);
    cudacpp::Runtime::sync();
    cudacpp::Runtime::memcpyDtoH(&numUniqueKeys, gpuValOffsets, sizeof(int));
    ++numUniqueKeys;
	
	//TODO tobe removed
	numUniqueKeys = 1;

    // 2 - C = gpuInputVals and
    // 3 - D = gpuValOffsets
    cudacpp::Runtime::sync();
    gpmrIntIntSorterFindOffsets(gpuInputKeys, gpuUniqueFlags, gpuValOffsets, gpuInputVals, gpuValOffsets, numKeys, numUniqueKeys);
    *numVals = reinterpret_cast<int * >(cudacpp::Runtime::mallocHost(numUniqueKeys * sizeof(int)));
    cudacpp::Runtime::sync();
    cudacpp::Runtime::memcpyDtoH(*numVals, gpuValOffsets, sizeof(int) * numUniqueKeys);
    cudacpp::Runtime::sync();

    // 4 - E = gpuUniqueFlags
    cudppConfig.algorithm  = CUDPP_SCAN;
    cudppConfig.op         = CUDPP_ADD; // ignored
    cudppConfig.datatype   = CUDPP_INT;
    cudppConfig.options    = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;
    cudppPlan(&planHandle, cudppConfig, numKeys, 1, numKeys * sizeof(int));
    cudppScan(planHandle, gpuUniqueFlags, gpuValOffsets, numKeys);
    cudppDestroyPlan(planHandle);
    cudacpp::Runtime::sync();
    *valOffsets = reinterpret_cast<int * >(cudacpp::Runtime::mallocHost(numUniqueKeys * sizeof(int)));
    cudacpp::Runtime::memcpyDtoH(*valOffsets, gpuUniqueFlags, sizeof(int) * numUniqueKeys);

    // 4 - F = gpuInputVals
    gpmrIntIntSorterSetCompactedKeys(gpuInputKeys, gpuUniqueFlags, gpuInputVals, numUniqueKeys);
    cudacpp::Runtime::memcpyDtoH(keys, gpuInputVals, sizeof(int) * numUniqueKeys);

    cudacpp::Runtime::free(gpuInputKeys);
    cudacpp::Runtime::free(gpuInputVals);
    cudacpp::Runtime::free(gpuUniqueFlags);
    cudacpp::Runtime::free(gpuValOffsets);

  }

  void IntIntSorter::executeOnCPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals)
  {
    int * iKeys     = reinterpret_cast<int * >(keys);
    int * iVals     = reinterpret_cast<int * >(vals);
    std::pair<int, int> * array = new std::pair<int, int>[numKeys];
    for (int i = 0; i < numKeys; ++i)
    {
      array[i].first = iKeys[i];
      array[i].second = iVals[i];
    }
    std::sort(array, array + numKeys);
    for (int i = 0; i < numKeys; ++i)
    {
      iKeys[i] = array[i].first;
      iVals[i] = array[i].second;
    }
    delete [] array;
    // intintsorter::quickSort<int, int>(iKeys, iVals, numKeys);
    numUniqueKeys = 1;
    for (int i = 1; i < numKeys; ++i)
    {
      if (iKeys[i - 1] != iKeys[i]) ++numUniqueKeys;
      if (iKeys[i - 1] >  iKeys[i])
      {
        printf("quick sort borked item[%d]=%d > item[%d]=%d.\n", i - 1, iKeys[i - 1], i, iKeys[i]);
        fflush(stdout);
      }
    }
    *valOffsets = reinterpret_cast<int * >(cudacpp::Runtime::mallocHost(numUniqueKeys * sizeof(int)));
    *numVals    = reinterpret_cast<int * >(cudacpp::Runtime::mallocHost(numUniqueKeys * sizeof(int)));

    int lastUniqueIndex = 0;
    int index = 0;
    for (int i = 1; i < numKeys; ++i)
    {
      if (iKeys[i - 1] != iKeys[i])
      {
        (*numVals)[index]    = i - lastUniqueIndex;
        (*valOffsets)[index] = lastUniqueIndex;
        ++index;
        lastUniqueIndex = i;
      }
    }
    (*numVals)[index]    = numKeys - lastUniqueIndex;
    (*valOffsets)[index] = lastUniqueIndex;
    for (int i = 0; i < numUniqueKeys; ++i)
    {
      iKeys[i] = iKeys[(*valOffsets)[i]];
    }
  }
}
