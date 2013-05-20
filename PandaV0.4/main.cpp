#include <mpi.h>
#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMapReduceJob.h>
#include <panda/IntIntSorter.h>
#include "WCMapper.h"
#include "WCReducer.h"
#include "MTRand.h"

#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>


int main(int argc, char ** argv)
{

  panda::MapReduceJob  * job = new panda::PandaMapReduceJob(argc, argv, true);
  int rank, size;
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //MPI_Comm_size(MPI_COMM_WORLD, &size);
  //printf("rank:%d size:%d\n", rank, size);

  job->setMessage (new panda::PandaMPIMessage(true));
  job->setMapper  (new WCMapper());
  job->setReducer (new WCReducer());
  job->setSorter  (new panda::IntIntSorter());

  //int provided;
  //MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided );

  if (rank == 0)
  {
    	char fn[256];
	char str[512];
		
	sprintf(fn,"/N/u/lihui/CUDA/github/panda/PandaV0.4/sample%d",gCommRank);
	int  chunk_size = 2048;
	char *chunk_data = (char *)malloc(sizeof(char)*(chunk_size+1000));
	FILE *wcfp;
	wcfp = fopen(fn, "r");
	if(wcfp == NULL)
		ShowLog("file:%s is broken",fn);
	else
		ShowLog("open file:%s",fn);
	
	const int NUM_ELEMENTS = 1;
	int total_len = 0;
	while(fgets(str,sizeof(str),wcfp) != NULL)
	{

		for (int i = 0; i < strlen(str); i++)
		str[i] = toupper(str[i]);
		strcpy((chunk_data + total_len),str);
		total_len += (int)strlen(str);
		if(total_len>chunk_size){
		ShowLog("add one input chunk");
		job->addInput(new panda::PreLoadedPandaChunk((char *)chunk_data, total_len, NUM_ELEMENTS ));
		MPI_Barrier(MPI_COMM_WORLD);
		job->execute();
		total_len=0;
		break;
		}//if

	}//while

  }//if
  MPI_Finalize();
  //delete job;
  return 0;
}
