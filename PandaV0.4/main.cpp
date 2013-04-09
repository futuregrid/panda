#include <mpi.h>
#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMapReduceJob.h>
#include <panda/IntIntSorter.h>
#include "WCMapper.h"
#include "WCReducer.h"
#include "MersenneTwister.h"

#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>


const int NUM_CENTERS = 1;
const int NUM_DIMS = 1;

int main(int argc, char ** argv)
{

  panda::MapReduceJob  * job = new panda::PandaMapReduceJob(argc, argv, true);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const float centers[10] = {0};
  
  job->setMessage (new panda::PandaMessage(sizeof(int), sizeof(int)));
  job->setMapper (new WCMapper());
  job->setReducer(new WCReducer());
  job->setSorter (new panda::IntIntSorter());

  if (rank == 0)
  {
	
    	char fn[256];
	char str[512];
	char strInput[1024];
	sprintf(fn,"sample.txt");
	int chunk_size = 512;
	char *chunk_data = (char *)malloc(sizeof(char)*(chunk_size+1000));
	FILE *wcfp;
	wcfp = fopen(fn, "r");
	double t1 = PandaTimer();
	const int NUM_ELEMENTS = 1;
	int total_len = 0;
	while(fgets(str,sizeof(str),wcfp) != NULL)
	{

		for (int i = 0; i < strlen(str); i++)
		str[i] = toupper(str[i]);
		strcpy((chunk_data + total_len),str);
		total_len += (int)strlen(str);
		if(total_len>chunk_size){
		printf("addInput\n");
		job->addInput(new panda::PreLoadedPandaChunk((char *)chunk_data, total_len, NUM_ELEMENTS ));
		job->execute();
		total_len=0;
		}//if

	}//while
	double t2 = PandaTimer();
	printf("Time:%f\n",t2-t1);	
  }//if (rank==0)

  delete job;
  return 0;
}
