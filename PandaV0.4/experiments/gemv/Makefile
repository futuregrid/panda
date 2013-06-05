#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  */
#*   MPI CUDA/MKL Matrix Multiplication       				      */
#*   CGL Indiana University                         			      */
#*   Author: Hui Li							      */
#*   Email: lihui@indiana.edu						      */
#*   04/20/2011                                				      */
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  */

.KEEP_STATE:

all: mpi

DFLAGS      =
MKLROOT     = /N/soft/intel/mkl/10.2.5.035/
MKLPATH     = /N/soft/intel/mkl/10.2.5.035/lib/em64t
MPIPATH     = /N/soft/intel/impi/4.0.0.028/lib64/
CUDAPATH    = /opt/cuda/lib64/

OPTFLAGS    = -O 
INCFLAGS    = -I$(MKLROOT)/include -I/N/soft/intel/impi/4.0.0.028/include64/
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)
LDFLAGS     = $(OPTFLAGS)
LIBS        = -lpthread
CORELIB = $(MKLPATH)/libmkl_core.a
IFACELIB = $(MKLPATH)/libmkl_intel_lp64.a
THREADINGLIB = $(MKLPATH)/libmkl_sequential.a
MKLLIBS =  $(IFACELIB) -Wl,--start-group $(THREADINGLIB) $(CORELIB) -Wl,--end-group

CC          = icc
MPICC       = mpiicc
FCC	    = fpp
.c.o:
	$(CC) $(CFLAGS) -c $<

H_FILES     = *.h

MPI_SRC     = pmm_mpi.c 

MPI_OBJ     = $(MPI_SRC:%.c=%.o)

pmm_mpi.o: pmm_mpi.c 
	$(MPICC) $(CFLAGS) -c $*.c
dgemm_cuda.o: dgemm_cuda.cu
	nvcc -c dgemm_cuda.cu -o dgemm_cuda.o

mpi: mpi_main
mpi_main: $(MPI_OBJ) dgemm_cuda.o
	$(MPICC) $(LDFLAGS) -o pmm_cuda_mkl dgemm_cuda.o $(MPI_OBJ) $(LIBS) $(MKLLIBS) -w -I$(MKLROOT)/include -L$(CUDAPATH) -L$(MPIPATH) -L$(MKLPATH) -lm -lcudart -lcublas

clean:
	rm -rf *.o pmm_cuda_mkl
