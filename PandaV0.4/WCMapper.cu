#include <panda/PandaGPUFunctions.h>


__global__ void wc(int dataSize, void *constgpuMemoryForChunk){
	
		int wsize = 0;
		char *start;
		char *p = (char *)constgpuMemoryForChunk;
		int *wc = (int *) malloc (sizeof(int));
		*wc = 1;
		int valSize = dataSize;
    
		while(1)
		{
			start = p;
			for(; *p>='A' && *p<='Z'; p++);

			*p='\0';
			++p;
			wsize=(int)(p-start);
			if (wsize>6){
				char *wkey = start;
				//GPUEmitMapOutput(wkey, wc, wsize, sizeof(int), d_g_state, map_task_idx);
			}//if
			valSize = valSize - wsize;
			if(valSize<=0)
				break;
		}//while
		__syncthreads();
}

void wcMapperExecute(
                         const int dataSize,
                         void * const gpuMemoryForChunk,
                         PandaGPUConfig & pandaGPUConfig,
                         cudaStream_t & stream)
{

  //TODO wcMapper

  //kmeansMapper      <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(gpuCenters, numCenters, coords, numElems, accumCenters, accumTotals);
  //kmeansAccumCenters<<<numCenters, NUM_BLOCKS,  0, stream>>>(pandaGPUConfig, accumTotals);
  //kmeansAccumCoords <<<numCenters, NUM_BLOCKS,  0, stream>>>(pandaGPUConfig, accumCenters);
	wc<<<1,1,0,stream>>>(dataSize, gpuMemoryForChunk);
}

//__device__ void gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){
//
//		int wsize = 0;
//		char *start;
//		char *p = (char *)VAL;
//		int *wc = (int *) malloc (sizeof(int));
//		*wc = 1;
//
//		while(1)
//		{
//			start = p;
//			for(; *p>='A' && *p<='Z'; p++);
//
//			*p='\0';
//			++p;
//			wsize=(int)(p-start);
//			if (wsize>6){
//				char *wkey = start;
//				GPUEmitMapOutput(wkey, wc, wsize, sizeof(int), d_g_state, map_task_idx);
//			}//if
//			valSize = valSize - wsize;
//			if(valSize<=0)
//				break;
//		}//while
//		
//		__syncthreads();
//		
//}//map2
