#include <mpi.h>

int main(int argc, char * argv[])
{

  int rank, size;
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("rank:%d size:%d\n",rank,size);

  return 0;

}
