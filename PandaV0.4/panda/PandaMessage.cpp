#include <mpi.h>
#include <panda/PandaMessageIORequest.h>
#include <panda/PandaMessage.h>
#include <cstring>

namespace panda
{

	PandaMPIMessage::PandaMPIMessage(const bool pCopySendData){
		copySendData  = pCopySendData;
	}//void

	PandaMPIMessage::~PandaMPIMessage(){
	
	}//void

	void PandaMPIMessage::setPnc(panda_node_context *pnc)
	{
		this->pnc = pnc;
	}//void

	void PandaMPIMessage::MsgInit(){
		Message::MsgInit();
		zeroReqs.resize(commSize);
		zeroCount[0]  = zeroCount[1] = zeroCount[2] = 0;
	}//void

	void PandaMPIMessage::MsgFinalize(){

	}//void

	void PandaMPIMessage::getFinalDataSize(int & keySize, int & valSize) const {}
	void PandaMPIMessage::getFinalDataSize(int & keySize, int & valSize, int & numKeys, int & numVals) const {}
	void PandaMPIMessage::getFinalData(void * keyStorage, void * valStorage) const {}
	void PandaMPIMessage::getFinalData(void * keyStorage, void * valStorage, int * keySizes, int * valSizes)const {}

	oscpp::AsyncIORequest * PandaMPIMessage::MsgFinish()
	{
    	return sendTo(commRank, NULL, NULL, NULL, -1, -1, -1);
	}//oscpp

  void PandaMPIMessage::run()
  {
    	//pnc -> received data 
	int finishedWorkers	= 0;
    	bool  * workerDone					= new bool[commSize];
    	bool  * recvingCount				= new bool[commSize];
    	int   * counts						= new int [commSize * 3];
    	int  ** keyRecv						= new int*[commSize];
    	int  ** valRecv						= new int*[commSize];
		int  ** keyPosKeySizeValPosValSize	= new int*[commSize];
	
	ShowLog("Start PandaMPIMessage Thread ...");
	if(workerDone==NULL)ShowLog("Error");
	if(recvingCount==NULL)ShowLog("Error");
	if(counts==NULL)ShowLog("Error");
	if(keyRecv==NULL)ShowLog("Error");
	if(valRecv==NULL)ShowLog("Error");
	if(keyPosKeySizeValPosValSize==NULL)ShowLog("Error");
	
    	MPI_Request * recvReqs  = new MPI_Request[commSize * 3];
	for (int i = 0; i < commSize; ++i)
    	{
      	workerDone[i]   = false;
      	recvingCount[i] = true;
      	keyRecv[i] 	= NULL;
      	valRecv[i] 	= NULL;
  	keyPosKeySizeValPosValSize[i] = NULL;
      	MPI_Irecv(counts + i * 3, 3, MPI_INT, i, 0, MPI_COMM_WORLD, recvReqs + i * 3);
    	}//for

	MPI_Barrier(MPI_COMM_WORLD);

	innerLoopDone 	= false;
   	int loopc 	= 0; 
	while (!innerLoopDone || finishedWorkers < commSize)
    	{
      	poll(finishedWorkers, workerDone, recvingCount, counts, keyRecv, valRecv, keyPosKeySizeValPosValSize, recvReqs);
	pollSends();
    	}//while

    	MPI_Waitall(commSize, &zeroReqs[0], MPI_STATUSES_IGNORE);

    	delete [] workerDone;
    	delete [] recvingCount;
    	delete [] counts;
    	delete [] keyRecv;
    	delete [] valRecv;
    	delete [] recvReqs;

	}
  
      oscpp::AsyncIORequest * PandaMPIMessage::sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
                                             const int keySize,
					     const int valSize) 
	  {
	  return NULL;
	  }

      	  oscpp::AsyncIORequest * PandaMPIMessage::sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
                                             int * const keySizes,
                                             int * const valSizes,
                                             const int numKeys,
					     const int numVals) 
	  {
	  return NULL;
	  }//oscpp

   //replace with ADIOS interface
   //write through MPI send and receive
   oscpp::AsyncIORequest * PandaMPIMessage::sendTo(const int rank,
                                                  void * const keys,
                                                  void * const vals,
						  int * const keyPosKeySizeValPosValSize,
                                                  const int keySize,
                                                  const int valSize,
						  const int maxlen)
  {
    PandaMessageIOData2 * data  = new PandaMessageIOData2;
    
    data->flag		= new volatile bool;
    data->waiting	= new volatile bool;
    *data->flag		= false;
    *data->waiting	= false;

    //write to disk for fault tolerance
    if (copySendData)
    {
      if (keySize > 0 && keys != NULL)
      {
	data->keysBuff = new char[keySize];
        memcpy(data->keysBuff, keys, keySize);
      }//if
      else
      {
        data->keysBuff = keys;
      }//else
      if (valSize > 0 && vals != NULL)
      {
        data->valsBuff = new char[valSize];
        memcpy(data->valsBuff, vals, valSize);
      }//if
      else
      {
        data->valsBuff = vals;
      }//else

      if (maxlen > 0 && keyPosKeySizeValPosValSize !=NULL)
      {
  	data->keyPosKeySizeValPosValSize = new int[4*maxlen];
  	memcpy(data->keyPosKeySizeValPosValSize, keyPosKeySizeValPosValSize, 4*maxlen*sizeof(int));
      }else
      {
  	data->keyPosKeySizeValPosValSize = keyPosKeySizeValPosValSize;
      }//if
    }
    else
    {
      data->keysBuff = keys;
      data->valsBuff = vals;
      data->keyPosKeySizeValPosValSize = keyPosKeySizeValPosValSize;
    }//else
    data->keyBuffSize = keySize;
    data->valBuffSize = valSize;
    data->rank = rank;

    if (rank == commRank)
    {
      data->counts[0] = maxlen;
      data->counts[1] = keySize;
      data->counts[2] = valSize;
      //data->done[0]   = data->done[1] = data->done[2] = data->done[3] = false;
    } //if
    else
    {
      data->counts[0] = maxlen;
      data->counts[1] = keySize;
      data->counts[2] = valSize;
      data->done[0]   = data->done[1] = data->done[2] = data->done[3] = false;
    } //else

    PandaMessageIORequest * req = new PandaMessageIORequest(data->flag, data->waiting, 
			4*data->counts[0]*sizeof(int) + data->counts[1] + data->counts[2]);
    data->cond = &req->condition();
    addDataLock.lock();
    needsToBeSent.push_back(data);
    addDataLock.unlock();
    return req;
  }


  bool PandaMPIMessage::pollUnsent()
  {
	keyvals_t *sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;
	int sorted_intermediate_keyvals_arr_len    = pnc->sorted_key_vals.sorted_keyvals_arr_len;
	
	char *sorted_keys_buff = (char *)pnc->sorted_key_vals.h_sorted_keys_shared_buff;
	char *sorted_vals_buff = (char *)pnc->sorted_key_vals.h_sorted_vals_shared_buff;

	int *keyPos;
	int *valPos;
	
	PandaMessageIOData2 * data = NULL;

    	addDataLock.lock();
    	if (!needsToBeSent.empty())
    	{
      	data = needsToBeSent.front();
      	needsToBeSent.pop_front();
    	}//if
    	addDataLock.unlock();

    	if (data == NULL){
		return false;
	}//if

    	if (data->rank == commRank)
    	{

  		if (data->counts[0] >= 0)
        {
	        //send to local host, be used for reduce task
			PandaAddRecvedBucket((char *)(data->keysBuff), (char *)(data->valsBuff), data->keyPosKeySizeValPosValSize, data->keyBuffSize, data->valBuffSize,data->counts[0]);
        }//if
        else
        {
			//finish the messager data->counts[0] == -1
			innerLoopDone = true;
			//The difference between SendTo(NULL,NULL,NULL,-1,-1,-1); is there is no need to creat data object
          for (int i = 0; i < commSize; ++i)
          {
			//the current messager is exiting and notify the other process prepare for exiting as well.
			//send zero count to ask receiver to exit the processing for peroformance issue
			//zeroCount;
			MPI_Isend(zeroCount, 3, MPI_INT, i, 0, MPI_COMM_WORLD, &zeroReqs[i]);
			//there is no need to push to pendingIO.
          } //for
      	}   //

        data->cond->lockMutex();
        if (*data->waiting) {
	  data->cond->broadcast();
	  //data is in buff, there is no need to waite
        } //if
      	*data->flag = true;
      	data->cond->unlockMutex();
      	delete data;

    	}//if
    	else
    	{
	  //send data asynchronizally, and put it in the pending task queue
	  if(data->counts[0] ==  0)
		ShowLog("! logic error here\n");

      	  MPI_Isend(data->counts,      3,    MPI_INT,     data->rank,  0,  MPI_COMM_WORLD, &data->reqs[0]);
	  MPI_Isend(data->keysBuff,    data->keyBuffSize, MPI_CHAR, data->rank, 0, MPI_COMM_WORLD, &data->reqs[1]);
	  MPI_Isend(data->valsBuff,    data->valBuffSize, MPI_CHAR, data->rank, 0, MPI_COMM_WORLD, &data->reqs[2]);
	  MPI_Isend(data->keyPosKeySizeValPosValSize, data->counts[0]*4, MPI_INT, data->rank, 0, MPI_COMM_WORLD, &data->reqs[3]);
      	  pendingIO.push_back(data);
    	} //else
    	return true;
  }

  void PandaMPIMessage::pollPending()
  {
    if (pendingIO.empty()) return;
    std::list<PandaMessageIOData2 * > newPending;
    for (std::list<PandaMessageIOData2 * >::iterator it = pendingIO.begin(); it != pendingIO.end(); ++it)
    {
      PandaMessageIOData2 * data = *it;
      int flag;
      MPI_Testall(4, data->reqs, &flag, data->stat);

      if (flag)
      {
        data->cond->lockMutex();
        if (*data->waiting) data->cond->broadcast();
        *data->flag = true;
        data->cond->unlockMutex();
        //TODO delete [] data->counts;
	//the data object has been sent out
	
        if (copySendData)
        {
		if (data->keysBuff != NULL) delete [] reinterpret_cast<char * >(data->keysBuff);
		if (data->valsBuff != NULL) delete [] reinterpret_cast<char * >(data->valsBuff);
	//TODO	if (data->keyPosKeySizeValPosValSize != NULL) delete [] reinterpret_cast<char *>(data->keyPosKeySizeValPosValSize);
        }//if
	//TODO
        //delete data;
      }//if
      else
      {
        newPending.push_back(data);
      }//else
    }//for
    pendingIO = newPending;
  }//void

	void PandaMPIMessage::pollSends()
	{
	
    	const int MAX_SENDS_PER_LOOP = 20;
    	int index = 0;
    	while (++index < MAX_SENDS_PER_LOOP && pollUnsent()) { }
    	  index = 0;
    	  pollPending();
	}//void

	void PandaMPIMessage::PandaAddRecvedBucket(char *keyRecv, char *valRecv, int *keyPosKeySizeValPosValSize, int keyBuffSize, int valBuffSize, int maxlen)
	{	
		//keyRecv[i], valRecv[i], keyPosKeySizeValPosValSize[i], counts[i * 3 + 1], counts[i * 3 + 2], counts[i * 3 + 2])
		
		char *newKeyRecv = new char[keyBuffSize];
		memcpy(newKeyRecv, keyRecv, keyBuffSize);
		pnc->recv_buckets.savedKeysBuff.push_back(newKeyRecv);

		char *newValRecv = new char[valBuffSize];
		memcpy(newValRecv, valRecv, valBuffSize);
		pnc->recv_buckets.savedValsBuff.push_back(newValRecv);
		
		int *keyPosArray = new int[maxlen];
		memcpy(keyPosArray, keyPosKeySizeValPosValSize, maxlen*sizeof(int));
		pnc->recv_buckets.keyPos.push_back(keyPosArray);

		int *keySizeArray = new int[maxlen];
		memcpy(keySizeArray, keyPosKeySizeValPosValSize+maxlen, maxlen*sizeof(int));
		pnc->recv_buckets.keySize.push_back(keySizeArray);

		int *valPosArray = new int[maxlen];
		memcpy(valPosArray, keyPosKeySizeValPosValSize+2*maxlen, maxlen*sizeof(int));
		pnc->recv_buckets.valPos.push_back(valPosArray);

		int *valSizeArray = new int[maxlen];
		memcpy(valSizeArray, keyPosKeySizeValPosValSize+3*maxlen, maxlen*sizeof(int));
		pnc->recv_buckets.valSize.push_back(valSizeArray);
		
		int *counts = new int[3];
		counts[0] = maxlen;
		counts[1] = keyBuffSize;
		counts[2] = valBuffSize;
		pnc->recv_buckets.counts.push_back(counts);
		
	}

	void PandaMPIMessage::poll(int & finishedWorkers,
                               bool * const workerDone,
                               bool * const recvingCount,
                               int *  const counts,
                               int ** keyRecv,
                               int ** valRecv,
			       int ** keyPosKeySizeValPosValSize,
                               MPI_Request * recvReqs)
  {
    pollSends();

    int flag;
    MPI_Status stat[3];

    for (int i = 0; i < commSize; ++i)
    {

      if (workerDone[i]) continue;
      //this is the head component of the communication protocal
      if (recvingCount[i])
      {
        MPI_Test(recvReqs + i * 3, &flag, stat);
        
	if (flag)
        {
          ShowLog(" - recv'd counts[0]:%d counts[1]:%d counts[2]:%d from [%d] ", 
			counts[i * 3 + 0], counts[i * 3 + 1], counts[i * 3 + 2], i); 
	  fflush(stdout);
          recvingCount[i] = false;
          if (counts[i * 3] == 0)  //zeroCount
          {
            workerDone[i] = true;
            ++finishedWorkers;
            ShowLog(" - recv'd 'finished' command from %d, now have %d finished workers.", i, finishedWorkers); 
	  
	    //fflush(stdout);
	    //TODO
	  
          }//if
          else
          {
	    ShowLog(" - recv'd counts[0]:%d counts[1]:%d counts[2]:%d from [%d] ", 
      		counts[i * 3 + 0], counts[i * 3 + 1], counts[i * 3 + 2], i); 

            keyRecv[i] = new int[(counts[i * 3 + 1] + sizeof(int) - 1) / sizeof(int)];
            valRecv[i] = new int[(counts[i * 3 + 2] + sizeof(int) - 1) / sizeof(int)];
	    keyPosKeySizeValPosValSize[i] = new int[4*counts[i * 3 + 0]];
	
            MPI_Irecv((char*)(keyRecv[i]), counts[i * 3 + 1], MPI_CHAR, i, 0, MPI_COMM_WORLD, recvReqs + i * 3 + 1);
            MPI_Irecv((char*)(valRecv[i]), counts[i * 3 + 2], MPI_CHAR, i, 0, MPI_COMM_WORLD, recvReqs + i * 3 + 2);
			MPI_Irecv(keyPosKeySizeValPosValSize[i], 4*counts[i * 3 + 0], MPI_INT, i, 0, MPI_COMM_WORLD, recvReqs + i * 3 + 0);
	
          }	//else
        }	//if flag
      }	  	//if recvingCount[i]
      else
      {

		//	recvingCount[i] != true
		//  all the data transfer have been completed.
		//	to test recvReqs
	
        MPI_Testall(3, recvReqs + i * 3, &flag, stat);

        if (flag)
        {
          //data have been received, add the data to reduce task																		    //maxlen
		  PandaAddRecvedBucket((char *)keyRecv[i], (char *)valRecv[i], keyPosKeySizeValPosValSize[i], counts[i * 3 + 1], counts[i * 3 + 2], counts[i * 3 + 0]);

          recvingCount[i] = true;
		  //the last signal zeroCount
          MPI_Irecv(counts + i * 3, 3, MPI_INT, i, 0, MPI_COMM_WORLD, recvReqs + i * 3);
	  
          //delete [] keyRecv[i];
          //delete [] valRecv[i];
		  //TODO delete keyPosKeySizeValPosValSize

        }//MPI_Testall
      }//else
    }
  }

	

	void PandaMessage::setPnc(panda_node_context *pnc){

	}

  bool PandaMessage::pollUnsent()
  {
    PandaMessageIOData * data = NULL;

    addDataLock.lock();
    if (!needsToBeSent.empty())
    {
      data = needsToBeSent.front();
      needsToBeSent.pop_front();
    }//if
    addDataLock.unlock();
    if (data == NULL) return false;

    if (data->rank == commRank)
    {
      if (data->keySize != -1)
      {
        privateAdd(data->keys, data->vals, data->keySize, data->valSize);
      }//if
      else
      {
        innerLoopDone = true;
        for (int i = 0; i < commSize; ++i)
        {
          MPI_Isend(zeroCount, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &zeroReqs[i]);
        }//for
      }//else
      data->cond->lockMutex();
      if (*data->waiting) data->cond->broadcast();
      *data->flag = true;
      data->cond->unlockMutex();
      delete data;
    }//if
    else
    {
      MPI_Isend(data->counts,             2, MPI_INT,  data->rank, 0, MPI_COMM_WORLD, &data->reqs[0]);
      MPI_Isend(data->keys,   data->keySize, MPI_CHAR, data->rank, 0, MPI_COMM_WORLD, &data->reqs[1]);
      MPI_Isend(data->vals,   data->valSize, MPI_CHAR, data->rank, 0, MPI_COMM_WORLD, &data->reqs[2]);
      pendingIO.push_back(data);
    }//else
    return true;
  }

  void PandaMessage::pollPending()
  {
    if (pendingIO.empty()) return;
    std::list<PandaMessageIOData * > newPending;
    for (std::list<PandaMessageIOData * >::iterator it = pendingIO.begin(); it != pendingIO.end(); ++it)
    {
      PandaMessageIOData * data = *it;
      int flag;

      MPI_Testall(3, data->reqs, &flag, data->stat);
      if (flag)
      {
        data->cond->lockMutex();
        if (*data->waiting) data->cond->broadcast();
        *data->flag = true;
        data->cond->unlockMutex();
        delete [] data->counts;
        if (copySendData)
        {
          if (data->keys != NULL) delete [] reinterpret_cast<char * >(data->keys);
          if (data->vals != NULL) delete [] reinterpret_cast<char * >(data->vals);
        }//if
        delete data;
      }//if
      else
      {
        newPending.push_back(data);
      }//else
    }//for
    pendingIO = newPending;
  }//void

  void PandaMessage::pollSends()
  {
    const int MAX_SENDS_PER_LOOP = 20;
    int index = 0;
    while (++index < MAX_SENDS_PER_LOOP && pollUnsent()) { }
    index = 0;
    pollPending();
  }//void

  void PandaMessage::poll(   int & finishedWorkers,
                             bool * const workerDone,
                             bool * const recvingCount,
                             int * const counts,
                             int ** keyRecv,
                             int ** valRecv,
                             MPI_Request * recvReqs)
  {
    pollSends();
    int flag;
    MPI_Status stat[2];
    for (int i = 0; i < commSize; ++i)
    {
      if (workerDone[i]) continue;
      if (recvingCount[i])
      {
        MPI_Test(recvReqs + i * 2, &flag, stat);
        if (flag)
        {
          // printf("%2d - recv'd counts %d and %d from %d.\n", commRank, counts[i * 2 + 0], counts[i * 2 + 1], i); fflush(stdout);
          recvingCount[i] = false;
          if (counts[i * 2] == 0)
          {
            workerDone[i] = true;
            ++finishedWorkers;
          }
          else
          {
            keyRecv[i] = new int[counts[i * 2 + 0] / sizeof(int)];
            valRecv[i] = new int[counts[i * 2 + 1] / sizeof(int)];
            MPI_Irecv(keyRecv[i], counts[i * 2 + 0], MPI_CHAR, i, 0, MPI_COMM_WORLD, recvReqs + i * 2 + 0);
            MPI_Irecv(valRecv[i], counts[i * 2 + 1], MPI_CHAR, i, 0, MPI_COMM_WORLD, recvReqs + i * 2 + 1);
          }//else
        }//if
      }//if
      else
      {
        MPI_Testall(2, recvReqs + i * 2, &flag, stat);
        if (flag)
        {
          privateAdd(keyRecv[i], valRecv[i], counts[i * 2 + 0], counts[i * 2 + 1]);
          recvingCount[i] = true;
          MPI_Irecv(counts + i * 2, 2, MPI_INT, i, 0, MPI_COMM_WORLD, recvReqs + i * 2);
          delete [] keyRecv[i];
          delete [] valRecv[i];
        }//MPI_Testall
      }//else
    }
  }

  void PandaMessage::grow(const int size, const int finalSize, int & finalSpace, char *& finals)
  {
    if (size + finalSize > finalSpace)
    {
      int newSpace = finalSpace * 2;
      while (size + finalSize > newSpace) newSpace *= 2;
      finalSpace = newSpace;
      char * temp = new char[finalSpace];
      memcpy(temp, finals, finalSize);
      delete [] finals;
      finals = temp;
    }//if
  }//void

  void PandaMessage::privateAdd(const void * const keys, const void * const vals, const int keySize, const int valSize)
  {

    grow(keySize, finalKeySize, finalKeySpace, finalKeys);
    grow(valSize, finalValSize, finalValSpace, finalVals);
    memcpy(finalKeys + finalKeySize, keys, keySize);
    memcpy(finalVals + finalValSize, vals, valSize);
    finalKeySize += keySize;
    finalValSize += valSize;

  }//void

  PandaMessage::PandaMessage(const int pSingleKeySize, const int pSingleValSize, const bool pCopySendData)
  {
    singleKeySize = pSingleKeySize;
    singleValSize = pSingleValSize;
    copySendData  = pCopySendData;
  }//PandaMessage

  PandaMessage::~PandaMessage()
  {

  }//PandaMessage

  oscpp::AsyncIORequest * PandaMessage::sendTo(const int rank,
                                                  void * const keys,
                                                  void * const vals,
                                                  const int keySize,
                                                  const int valSize)
  {
    PandaMessageIOData * data  = new PandaMessageIOData;
    data->flag = new volatile bool;
    data->waiting = new volatile bool;
    *data->flag = false;
    *data->waiting = false;
    if (copySendData)
    {
      if (keySize > 0 && keys != NULL)
      {
        data->keys = new char[keySize];
        memcpy(data->keys, keys, keySize);
      }
      else
      {
        data->keys = keys;
      }
      if (valSize > 0 && vals != NULL)
      {
        data->vals = new char[valSize];
        memcpy(data->vals, vals, valSize);
      }
      else
      {
        data->vals = vals;
      }
    }
    else
    {
      data->keys = keys;
      data->vals = vals;
    }
    data->keySize = keySize;
    data->valSize = valSize;
    data->rank = rank;

    if (rank == commRank)
    {
      data->counts = NULL;
    }
    else
    {
      data->counts = new int[2];
      data->counts[0] = keySize;
      data->counts[1] = valSize;
      data->done[0] = data->done[1] = data->done[2] = false;
    }
    PandaMessageIORequest * req = new PandaMessageIORequest(data->flag, data->waiting, data->keySize + data->valSize);
    data->cond = &req->condition();

    addDataLock.lock();
    needsToBeSent.push_back(data);
    addDataLock.unlock();
    return req;
  }
  
  oscpp::AsyncIORequest * PandaMessage::sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
											 int * const keyPosKeySizeValPosValSize,
                                             const int keySize,
                                             const int valSize,
											 const int maxlen)
  {
	  return NULL;
  }

  oscpp::AsyncIORequest * PandaMessage::sendTo(const int rank,
                                                  void * const keys,
                                                  void * const vals,
                                                  int * const keySizes,
                                                  int * const valSizes,
                                                  const int numKeys,
                                                  const int numVals)
  {
    return NULL;
  }//oscpp

  void PandaMessage::MsgInit()
  {
    Message::MsgInit();
    zeroReqs.resize(commSize);
    zeroCount[0]  = zeroCount[1] = 0;
    finalKeySpace = 1048576;
    finalValSpace = 1048576;
    finalKeySize  = 0;
    finalValSize  = 0;
    finalKeys     = new char[finalKeySpace];
    finalVals     = new char[finalValSpace];
  }//void

  void PandaMessage::MsgFinalize()
  {
    delete [] finalKeys;
    delete [] finalVals;
  }//void
  
  void PandaMessage::run()
  {
    //run 
    int finishedWorkers = 0;
    bool  * workerDone      = new bool[commSize];
    bool  * recvingCount    = new bool[commSize];
    int   * counts          = new int [commSize * 2];
    int  ** keyRecv         = new int*[commSize];
    int  ** valRecv         = new int*[commSize];
    MPI_Request * recvReqs  = new MPI_Request[commSize * 2];

    for (int i = 0; i < commSize; ++i)
    {
      workerDone[i] = false;
      recvingCount[i] = true;
      keyRecv[i] = NULL;
      valRecv[i] = NULL;
      MPI_Irecv(counts + i * 2, 2, MPI_INT, i, 0, MPI_COMM_WORLD, recvReqs + i * 2);
    }//for

    innerLoopDone = false;
    while (!innerLoopDone || finishedWorkers < commSize)
    {
      poll(finishedWorkers, workerDone, recvingCount, counts, keyRecv, valRecv, recvReqs);
      pollSends();
    }//while
    MPI_Waitall(commSize, &zeroReqs[0], MPI_STATUSES_IGNORE);

    delete [] workerDone;
    delete [] recvingCount;
    delete [] counts;
    delete [] keyRecv;
    delete [] valRecv;
    delete [] recvReqs;

  }//void

  oscpp::AsyncIORequest * PandaMessage::MsgFinish()
  {
    return sendTo(commRank, NULL, NULL, -1, -1);
  }//oscpp

  void PandaMessage::getFinalDataSize(int & keySize, int & valSize) const
  {
    keySize = finalKeySize;
    valSize = finalValSize;
  }//void

  void PandaMessage::getFinalDataSize(int & keySize, int & valSize, int & numKeys, int & numVals) const
  { // not used
  }//void
  void PandaMessage::getFinalData(void * keyStorage, void * valStorage) const
  {
    memcpy(keyStorage, finalKeys, finalKeySize);
    memcpy(valStorage, finalVals, finalValSize);
  }//void
  void PandaMessage::getFinalData(void * keyStorage, void * valStorage, int * keySizes, int * valSizes) const
  { // not used
  }//void
}
