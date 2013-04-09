#include <panda/Message.h>
#include <mpi.h>

namespace panda
{
  Message::Message() : commSize(-1), commRank(-1)
  {

  }//message

  Message::~Message()
  {
  }//message

  void Message::init()
  {
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  }//void
}
