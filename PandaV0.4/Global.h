/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: Panda.h 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
	
 */

/*
#ifdef WIN32 
#include <windows.h> 
#endif 
#include <pthread.h>
*/


#ifndef __PANDA_GLOBAL_H__
#define __PANDA_GLOBAL_H__

#include "cudacpp/Array.h"
#include "cudacpp/ChannelFormatDescriptor.h"
#include "cudacpp/DeviceProperties.h"
#include "cudacpp/Error.h"
#include "cudacpp/Event.h"
#include "cudacpp/Kernel.h"
#include "cudacpp/KernelConfiguration.h"
#include "cudacpp/KernelParameters.h"
#include "cudacpp/Runtime.h"
#include "cudacpp/Stream.h"
#include "cudacpp/myString.h"

#include "cudacpp/Vector2.h"
#include "cudacpp/Vector3.h"
#include "cudacpp/Vector4.h"


#include "panda/Message.h"
#include "panda/Chunk.h"
#include "panda/Combiner.h"
#include "panda/EmitConfiguration.h"
#include "panda/PandaMessageIORequest.h"
#include "panda/PandaChunk.h"

#include "panda/PandaMapReduceJob.h"
#include "panda/PandaCPUConfig.h"
#include "panda/PandaGPUAtomicFunctions.h"


#include "panda/PandaGPUGridFunctions.h"
#include "panda/PandaGPUThreadFunctions.h"
#include "panda/IntIntSorter.h"
#include "panda/IntIntRoundRobinPartitioner.h"
#include "panda/ItemCollection.h"
#include "panda/Mapper.h"
#include "panda/MapReduceJob.h"
#include "panda/PartialSorter.h"
#include "panda/Partitioner.h"
#include "panda/PreLoadedPandaChunk.h"
#include "panda/Reducer.h"
#include "panda/SerializedItemCollection.h"
#include "panda/Sorter.h"
#include "panda/VectorOps.h"
//
#include "oscpp/AsyncFileReader.h"
#include "oscpp/AsyncIORequest.h"
#include "oscpp/Closure.h"
#include "oscpp/Condition.h"
#include "oscpp/GenericAsyncIORequest.h"
#include "oscpp/Mutex.h"
#include "oscpp/Runnable.h"
#include "oscpp/Thread.h"
#include "oscpp/Win32AsyncIORequest.h"

#endif