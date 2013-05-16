# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.42 						  5/16/2013 */
#    							  lihui@indiana.edu */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: panda

OPTFLAGS    = -O 
INCFLAGS    = -I. -I/opt/openmpi-1.4.3-gnu/include/
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0 
NVCCFLAGS   = $(CFLAGS) --ptxas-options=-v -arch=sm_20
LDFLAGS     = -L/opt/openmpi-1.4.3-gnu/lib/
LIBS        = -lmpi  -lmpi_cxx -lpthread -lcutil_x86_64 -lcudart

#  note:
#  gcc usess -fopenmp
#  icc uses -opnemp

OMPFLAGS    = -fopenmp
CC          = mpicxx
MPICC       = mpicxx
NVCC        = nvcc

NVCCFLAGS  += -Xcompiler -fopenmp
INCFLAGS   += -I/opt/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/ 
INCFLAGS   += -I./include -I./
LDFLAGS    += -L/opt/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

OS_CPP_FILES 	:= $(wildcard oscpp/*.cpp)
PANDA_CPP_FILES := $(wildcard panda/*.cpp)
CUDA_CPP_FILES 	:= $(wildcard cudacpp/*.cpp)
CUDA_CU_FILES 	:= $(wildcard cuda/*.cu)

OS_H_FILES 	:= $(wildcard include/oscpp/*.h)
PANDA_H_FILES 	:= $(wildcard include/panda/*.h)
CUDA_H_FILES 	:= $(wildcard include/cudacpp/*.h)
H_FILES 	:= $(wildcard *.h)

OS_OBJ_FILES 	:= $(addprefix obj/,$(notdir $(OS_CPP_FILES:.cpp=.o)))
PANDA_OBJ_FILES := $(addprefix obj/,$(notdir $(PANDA_CPP_FILES:.cpp=.o)))
CUDA_OBJ_FILES 	:= $(addprefix obj/,$(notdir $(CUDA_CPP_FILES:.cpp=.o)))
CU_OBJ_FILES 	:= $(addprefix cuobj/,$(notdir $(CUDA_CU_FILES:.cu=.o)))
CUDPP_OBJ_FILES := obj/cudpp.o

WC_OBJ_FILES	:= obj/main.o obj/WCMapper.o obj/WCReducer.o
WC_OBJ_CU_FILES	:= cuobj/WCMapper.o cuobj/WCReducer.o

panda: panda_main
panda_main: $(WC_OBJ_CU_FILES) $(WC_OBJ_FILES) $(OS_OBJ_FILES) $(PANDA_OBJ_FILES) \
		$(CUDA_OBJ_FILES) $(CU_OBJ_FILES) $(CUDPP_OBJ_FILES)
	g++ $(LIBS) $(LDFLAGS) -o $@ $^

obj/%.o: oscpp/%.cpp $(OS_H_FILES)	
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: panda/%.cpp $(PANDA_H_FILES) $(H_FILES)
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: cudacpp/%.cpp $(CUDA_H_FILES)
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

cuobj/%.o: cuda/%.cu $(CUDA_H_FILES) $(H_FILES)
	nvcc $(LIBS) $(NVCCFLAGS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: ./%.cpp $(OS_H_FILES) $(PANDA_H_FILES) $(CUDA_H_FILES) $(H_FILES)
	g++ $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

cuobj/WCMapper.o: ./WCMapper.cu $(PANDA_H_FILES)
	nvcc $(LIBS) $(NVCCFLAGS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

cuobj/WCReducer.o: ./WCReducer.cu $(PANDA_H_FILES)
	nvcc $(LIBS) $(NVCCFLAGS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/cudpp.o:	cudpp/cudpp.cpp include/cudpp/cudpp.h
	g++ $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

clean:
	rm -rf obj/*.o cuobj/*.o panda_main 
