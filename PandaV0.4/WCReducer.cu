




void wcReducerExecute(const int numKeys,
                          const int   * const numVals,
                          const int   * const oldKeys,
                                int   * const newKeys,
                          const void  * const oldVals,
                                void  * const newVals,
                          cudaStream_t & stream)
{
  //TODO
  //kmeansReducerKernel<<<1, numKeys, 0, stream>>>(numKeys, numVals, oldKeys, newKeys, oldVals, newVals);
}
