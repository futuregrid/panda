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
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	job->setMessage (new panda::PandaMPIMessage(true));
	job->setMapper (new WCMapper());
	job->setReducer(new WCReducer());
	job->setSorter (new panda::IntIntSorter());

	if (argc != 7)
	{
		printf("Panda C-means\n");
		printf("usage: %s [numPt]  [cpu/gpu ratio] [numMapperPerCPU][numMapperPerGPU] [numGPU] [numCPU]\n", argv[0]);
	}//if
	
	int numPt = atoi(argv[1]);
	int dim = 10;//atoi(argv[2]);
	int K = 10;//atoi(argv[3]);
	float ratio = atof(argv[2]);
	int numMapperCPU = atoi(argv[3]);
	int numMapperGPU = atoi(argv[4]);
		
	int maxIter = 10;//atoi(argv[5]);

	numMapperGPU = 1;
	numMapperCPU = 1;

	int num_gpu_core_groups = 0;
	int num_gpu_card_groups = 1;
	int num_cpus_groups = 0;

	panda_context *panda = CreatePandaContext();
	panda->num_gpu_core_groups = num_gpu_core_groups;
	panda->num_gpu_card_groups = num_gpu_card_groups;
	panda->num_cpus_groups = num_cpus_groups;

	panda->cpu_ratio = 0.0;
	panda->gpu_card_ratio = 1.0;
	panda->gpu_core_ratio = 0.0;

	ShowLog("numPt:%d	dim:%d	K:%d	numMapperGPU:%d numMapperCPU:%d	maxIter:%d",numPt,dim,K,numMapperGPU,numMapperCPU,maxIter);
	float* h_points = GenPointsFloat(numPt, dim);
	float* h_cluster = GenInitCentersFloat(h_points, numPt, dim, K);
	
	int numgpus = -1;
	cudaGetDeviceCount(&numgpus);

	if (num_gpu_core_groups > numgpus)		num_gpu_core_groups = numgpus;
	if (num_gpu_card_groups > numgpus-num_gpu_core_groups) num_gpu_card_groups = numgpus-num_gpu_core_groups;
			
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpu_core_groups+num_gpu_card_groups+num_cpus_groups));
	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*(num_gpu_core_groups+num_gpu_card_groups+num_cpus_groups));
		
	int gpuCoreWorkLoad = int(numPt * panda->gpu_core_ratio);
	int gpuCardWorkLoad = int(numPt * panda->gpu_card_ratio);
	int cpuWorkLoad = int(numPt * panda->cpu_ratio);

	gpuCardWorkLoad = numPt;
	gpuCoreWorkLoad = 0;
	cpuWorkLoad = 0;

	ShowLog("numPt:%d cpuWorkload:%d gpuCoreWorkload:%d gpuCardWorkload:%d", numPt, cpuWorkLoad, gpuCoreWorkLoad, gpuCardWorkLoad);
	int global_dev_id = 0;

	for (int dev_id=0; dev_id<num_gpu_core_groups; dev_id++, global_dev_id++){
		
		job_configuration *gpu_job_conf = CreateJobConf();
		gpu_job_conf->num_gpu_core_groups = num_gpu_core_groups;
		gpu_job_conf->num_mappers = numMapperGPU;
		gpu_job_conf->auto_tuning = false;
		gpu_job_conf->auto_tuning_sample_rate = -1;//sample_rate;
		gpu_job_conf->iterative_support = false;

		int tid = global_dev_id;		
		float* d_points	=	NULL;
		float* d_cluster =	NULL;
		int* d_clusterId =	NULL;
		
		float* d_tempClusters = NULL;
		float* d_tempDenominators = NULL;
		
		if (tid<numgpus)
		checkCudaErrors(cudaSetDevice(tid));

		checkCudaErrors(cudaMalloc((void**)&d_points, numPt*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_points, h_points, numPt*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_clusterId, numPt*sizeof(int)));
		checkCudaErrors(cudaMemset(d_clusterId, 0, numPt*sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_cluster, K*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_cluster, h_cluster, K*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_tempClusters,K*dim*numMapperGPU*sizeof(float)));
		checkCudaErrors(cudaMemset(d_tempClusters, 0, sizeof(float)*K*dim*numMapperGPU));
		checkCudaErrors(cudaMalloc((void**)&d_tempDenominators,numMapperGPU * K * sizeof(float)));
		
		checkCudaErrors(cudaMemset(d_tempDenominators, 0, sizeof(float)*K*numMapperGPU));
		thread_info[global_dev_id].tid = global_dev_id;
		thread_info[global_dev_id].device_type = GPU_CORE_ACC;
		
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, dev_id);
		ShowLog("Configure Device ID:%d: Device Name:%s", dev_id, gpu_dev.name);
		thread_info[dev_id].device_name = gpu_dev.name;
				
		KM_VAL_T val;
		val.d_Points = d_points;
		val.d_Clusters = d_cluster;
		
		KM_KEY_T key;
		key.dim = dim;
		key.K = K;
		
		int numPtPerGPU = gpuCoreWorkLoad/num_gpu_core_groups;
		int start = dev_id*numPtPerGPU;
		int end = start+numPtPerGPU;
		if (dev_id==num_gpu_core_groups-1)
			end = gpuCoreWorkLoad;
		
		int numPtPerMap = (end-start)/numMapperGPU;
		ShowLog("GPU core numPtPerMap:%d startPt:%d  endPt:%d numPt:%d gpuCoreWorkLoad:%d",numPtPerMap,start,end,numPt,gpuCoreWorkLoad);

		int start_i,end_i;
		start_i = start;
		for (int j = 0; j < numMapperGPU; j++)
		{	
			end_i = start_i + numPtPerMap;
			if ( j < (end-start)%numMapperGPU)
				end_i++;
			
			key.start = start_i;
			key.end = end_i;
			key.global_map_id = dev_id*numMapperGPU+j;
			key.local_map_id = j;

			val.d_Points = d_points;
			val.d_tempDenominators = d_tempDenominators;
			val.d_tempClusters = d_tempClusters;

			AddPandaTask(gpu_job_conf, &key, &val, sizeof(KM_KEY_T), sizeof(KM_VAL_T));
			start_i = end_i;
		}//for
		thread_info[dev_id].job_conf = gpu_job_conf;
		thread_info[dev_id].device_type = GPU_CORE_ACC;
	}//for

	for (int dev_id = 0; dev_id < num_gpu_card_groups; dev_id++, global_dev_id++){

		job_configuration *gpu_job_conf = CreateJobConf();
		gpu_job_conf->num_gpu_card_groups = num_gpu_card_groups;
		gpu_job_conf->num_mappers = numgpus;

		gpu_job_conf->auto_tuning = false;
		gpu_job_conf->auto_tuning_sample_rate = -1;//sample_rate;
		gpu_job_conf->iterative_support = false;

		int tid = global_dev_id;	
		KM_VAL_T val;
		val.d_Points = h_points;
		val.d_Clusters = h_cluster;

		KM_KEY_T key;
		key.dim = dim;
		key.K = K;
		
		int numPtPerGPU = gpuCoreWorkLoad/num_gpu_card_groups;
		int start = dev_id*numPtPerGPU;
		int end = start+numPtPerGPU;
		if (dev_id==num_gpu_card_groups-1)
			end = gpuCardWorkLoad;
		
		int numPtPerMap = (end-start);
		ShowLog("GPU card numPtPerMap:%d startPt:%d  endPt:%d numPt:%d gpuCardWorkLoad:%d",numPtPerMap,start,end,numPt,gpuCardWorkLoad);

		float* h_tempClusters = NULL;
		float* h_tempDenominators = NULL;
		h_tempClusters = (float *)malloc(K*dim*numMapperGPU*sizeof(float));
		h_tempDenominators = (float *)malloc(numMapperGPU*K*sizeof(float));

		int start_i,end_i;
		start_i = start;

		for (int j = 0; j < 1; j++)
		{	
			end_i = start_i + numPtPerMap;
			key.start = start_i;
			key.end = end_i;
			key.global_map_id = global_dev_id;
			key.local_map_id = j;

			val.d_tempDenominators = h_tempDenominators;
			val.d_tempClusters = h_tempClusters;

			AddPandaTask(gpu_job_conf, &key, &val, sizeof(KM_KEY_T), sizeof(KM_VAL_T));
			start_i = end_i;
		}//for
		thread_info[dev_id].job_conf = gpu_job_conf;
		thread_info[dev_id].device_type = GPU_CARD_ACC;

	}//for

	for (int dev_id = 0; dev_id < num_cpus_groups; dev_id++, global_dev_id++){

		job_configuration *cpu_job_conf = CreateJobConf();
		cpu_job_conf->num_cpus_groups = num_cpus_groups;
		cpu_job_conf->num_cpus_cores = getCPUCoresNum();
		cpu_job_conf->auto_tuning_sample_rate = -1;
		cpu_job_conf->iterative_support = false;
		cpu_job_conf->local_combiner = false;

		int tid = dev_id;		

		
		float* h_tempClusters = NULL;
		float* h_tempDenominators = NULL;

		h_tempClusters = (float *)malloc(K*dim*numMapperGPU*sizeof(float));
		h_tempDenominators = (float *)malloc(numMapperGPU*K*sizeof(float));
		
		thread_info[dev_id].device_name = "CPU";
			
		KM_VAL_T val;
		
		val.d_Points = h_points;
		val.d_Clusters = h_cluster;
		
		KM_KEY_T key;
		key.dim = dim;
		key.K = K;
		
		int numPtPerCPU = cpuWorkLoad/num_cpus_groups;
		int start = gpuCoreWorkLoad+ (dev_id-num_gpu_core_groups)*numPtPerCPU;
		int end = start+numPtPerCPU;

		if (dev_id==num_gpu_core_groups+num_cpus_groups-1)
			end = numPt;
		
		int numPtPerMap = (end-start)/numMapperCPU;
		ShowLog("CPU numPtPerMap:%d startPt:%d  endPt:%d numPt:%d cpuWorkLoad:%d",numPtPerMap,start,end,numPt,cpuWorkLoad);

		int start_i, end_i;
		start_i = start;

		if (cpuWorkLoad>0)
		for (int j = 0; j < numMapperCPU; j++)
		{	
			end_i = start_i + numPtPerMap;
			if ( j < (end-start)%numMapperCPU )
				end_i++;

			if(end_i==start_i)		break;

			key.start = start_i;
			key.end = end_i;
			key.global_map_id = dev_id*numMapperCPU+j;
			key.local_map_id = j;

			val.d_tempDenominators = h_tempDenominators;
			val.d_tempClusters = h_tempClusters;

			AddPandaTask(cpu_job_conf, &key, &val, sizeof(KM_KEY_T), sizeof(KM_VAL_T));
			start_i = end_i;
		}//for
		
		thread_info[dev_id].job_conf = cpu_job_conf;
		thread_info[dev_id].device_type = CPU_ACC;
		thread_info[dev_id].tid = dev_id;
		
	}//for

	//todo here
	/*if (rank == 0)
	{
    char fn[256];
	char str[512];
	char strInput[1024];
	sprintf(fn,"sample.txt");
	int  chunk_size = 1024;
	char *chunk_data = (char *)malloc(sizeof(char)*(chunk_size+1000));
	FILE *wcfp;
	wcfp = fopen(fn, "r");

	const int NUM_ELEMENTS = 1;
	int total_len = 0;
	while(fgets(str,sizeof(str),wcfp) != NULL)
	{
		for (int i = 0; i < strlen(str); i++)
		str[i] = toupper(str[i]);
			
		strcpy((chunk_data + total_len),str);
		total_len += (int)strlen(str);
		
		if(total_len>chunk_size){*/
		job->addInput(new panda::PreLoadedPandaChunk((char *)chunk_data, total_len, NUM_ELEMENTS ));
		job->execute();
		total_len=0;
	//	}//if
	//}//while
	//}//if

	delete job;
	return 0;
}
