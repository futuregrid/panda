#include <cudpp/cudpp.h>

//#define DLL_EXPORT
//CUDPP_DLL
CUDPPResult cudppPlan(CUDPPHandle        *planHandle, 
                      CUDPPConfiguration config, 
                      size_t             n, 
                      size_t             rows, 
					  size_t             rowPitch){
CUDPPResult res = CUDPP_SUCCESS;
return res;
}


//CUDPP_DLL
CUDPPResult cudppDestroyPlan(CUDPPHandle plan){
CUDPPResult res = CUDPP_SUCCESS;
return res;
}

// Scan and sort algorithms
//CUDPP_DLL
CUDPPResult cudppScan(CUDPPHandle planHandle,
                      void        *d_out, 
                      const void  *d_in, 
					  size_t      numElements){
CUDPPResult res= CUDPP_SUCCESS;
return res;
}

//CUDPP_DLL
CUDPPResult cudppMultiScan(CUDPPHandle planHandle,
                           void        *d_out, 
                           const void  *d_in, 
                           size_t      numElements,
						   size_t      numRows){
CUDPPResult res= CUDPP_SUCCESS;
return res;
}

//CUDPP_DLL
CUDPPResult cudppSegmentedScan(CUDPPHandle        planHandle,
                               void               *d_out, 
                               const void         *d_idata,
                               const unsigned int *d_iflags,
							   size_t             numElements){
CUDPPResult res= CUDPP_SUCCESS;
return res;
}

//CUDPP_DLL
CUDPPResult cudppCompact(CUDPPHandle        planHandle,
                         void               *d_out, 
                         size_t             *d_numValidElements,
                         const void         *d_in, 
                         const unsigned int *d_isValid,
						 size_t             numElements){
CUDPPResult res= CUDPP_SUCCESS;
return res;
}

//CUDPP_DLL
CUDPPResult cudppSort(CUDPPHandle planHandle,
                      void        *d_keys,                                          
                      void        *d_values,                                                                       
                      int         keybits,
					  size_t      numElements){
CUDPPResult res = CUDPP_SUCCESS;
return res;
}

// Sparse matrix allocation
//CUDPP_DLL
CUDPPResult cudppSparseMatrix(CUDPPHandle        *sparseMatrixHandle, 
                              CUDPPConfiguration config, 
                              size_t             n, 
                              size_t             rows, 
                              const void         *A,
                              const unsigned int *h_rowIndices,
							  const unsigned int *h_indices){

	CUDPPResult res= CUDPP_SUCCESS;
	return res;

}

//CUDPP_DLL
CUDPPResult cudppDestroySparseMatrix(CUDPPHandle sparseMatrixHandle){

	CUDPPResult res= CUDPP_SUCCESS;
	return res;

}

// Sparse matrix-vector algorithms
//CUDPP_DLL
CUDPPResult cudppSparseMatrixVectorMultiply(CUDPPHandle sparseMatrixHandle,
                                            void        *d_y,
											const void  *d_x){

	CUDPPResult res= CUDPP_SUCCESS;
	return res;

}

//random number generation algorithms
//CUDPP_DLL
CUDPPResult cudppRand(CUDPPHandle planHandle,void * d_out, size_t numElements){

	CUDPPResult res= CUDPP_SUCCESS;
	return res;

}

//CUDPP_DLL
CUDPPResult cudppRandSeed(const CUDPPHandle planHandle, unsigned int seed){
	
	CUDPPResult res= CUDPP_SUCCESS;
	return res;

}//CUDPPResult