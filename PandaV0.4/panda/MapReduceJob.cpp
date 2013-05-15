#include <mpi.h>
#include <panda/Message.h>
#include <panda/Chunk.h>
#include <panda/Combiner.h>
#include <panda/PandaGPUConfig.h>
#include <panda/Mapper.h>
#include <panda/MapReduceJob.h>
#include <panda/Partitioner.h>
#include <panda/PartialReducer.h>
#include <panda/Reducer.h>
#include <panda/SerializedItemCollection.h>
#include <panda/Sorter.h>

#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>

#include <oscpp/Condition.h>
#include <oscpp/Runnable.h>
#include <oscpp/Thread.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <list>
#include <string>
#include <utility>
#include <vector>


#ifdef _WIN32
#include <windows.h> 
  #define popen _popen
  #define pclose _pclose
#endif

namespace panda
{

  void MapReduceJob::setDevice()
  {
#ifdef _WIN32
    FILE * fp = popen("hostname.exe", "r");
#else
    FILE * fp = popen("/bin/hostname", "r");
#endif

    char buf[1024];
    if (fgets(buf, 1023, fp) == NULL) strcpy(buf, "localhost");
    pclose(fp);
    std::string host = buf;
    host = host.substr(0, host.size() - 1);
    strcpy(buf, host.c_str());

    int devCount = cudacpp::Runtime::getDeviceCount();
    printf("GPU device count:%d on master node:%s\n",devCount,buf);

	if (commRank == 0)
    {
      std::map<std::string, std::vector<int> > hosts;
      std::map<std::string, int> devCounts;
      MPI_Status stat;
      MPI_Request req;

      hosts[buf].push_back(0);
      devCounts[buf] = devCount;
      for (int i = 1; i < commSize; ++i)
      {
        MPI_Recv(buf, 1024, MPI_CHAR, i, 0, MPI_COMM_WORLD, &stat);
        MPI_Recv(&devCount, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);

        // check to make sure each process on each node reports the same number of devices.
        hosts[buf].push_back(i);
        if (devCounts.find(buf) != devCounts.end())
        {
          if (devCounts[buf] != devCount)
          {
            printf("Error, device count mismatch %d != %d on %s\n", devCounts[buf], devCount, buf); fflush(stdout);
          }
        }
        else devCounts[buf] = devCount;
      }
      // check to make sure that we don't have more jobs on a node than we have GPUs.
      for (std::map<std::string, std::vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it)
      {
        if (it->second.size() > static_cast<unsigned int>(devCounts[it->first]))
        {
          printf("Error, more jobs running on '%s' than devices - %d jobs > %d devices.\n",
                 it->first.c_str(), static_cast<int>(it->second.size()), devCounts[it->first]);
          fflush(stdout);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }
#if 1 // print out the configuration
      for (std::map<std::string, std::vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it)
      {
        printf("%s - %d\n", it->first.c_str(), devCounts[it->first]);
        for (unsigned int i = 0; i < it->second.size(); ++i) printf("  %d\n", it->second[i]);
      }
      fflush(stdout);
#endif

      // send out the device number for each process to use.
      MPI_Irecv(&deviceNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
      for (std::map<std::string, std::vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it)
      {
        for (unsigned int i = 0; i < it->second.size(); ++i)
        {
          int devID = i;
          MPI_Send(&devID, 1, MPI_INT, it->second[i], 0, MPI_COMM_WORLD);
        }
      }
      MPI_Wait(&req, &stat);
    }
    else
    {
      // send out the hostname and device count for your local node, then get back the device number you should use.
      MPI_Status stat;
      MPI_Send(buf, strlen(buf) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&devCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(&deviceNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
    }

#if 1 // print out stuff
    MPI_Barrier(MPI_COMM_WORLD);
    printf("%d %s - using device %d (getDevice returns %d).\n", commRank, host.c_str(), deviceNum, cudacpp::Runtime::getDevice()); fflush(stdout);
#endif
    cudacpp::Runtime::setDevice(deviceNum);
    MPI_Barrier(MPI_COMM_WORLD);
  }//MPI_Barrier(MPI_COMM_WORLD);

  void MapReduceJob::collectTimings()
  {
  }//void

  MapReduceJob::MapReduceJob(int & argc, char **& argv)
    : messager(NULL), combiner(NULL), mapper(NULL), partitioner(NULL), partialReducer(NULL), sorter(NULL), reducer(NULL), commRank(-1), commSize(-1), deviceNum(-1)
  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    setDevice();
  }

  MapReduceJob::~MapReduceJob()
  {
    if (messager          != NULL) delete messager;
    if (combiner        != NULL) delete combiner;
    if (mapper          != NULL) delete mapper;
    if (partitioner     != NULL) delete partitioner;
    if (partialReducer  != NULL) delete partialReducer;
    if (sorter          != NULL) delete sorter;
    if (reducer         != NULL) delete reducer;
    MPI_Finalize();

  }//MapReduceJob
}
