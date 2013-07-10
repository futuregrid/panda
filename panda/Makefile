# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.43 						  7/4/2013  */
#    							  lihui@indiana.edu */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: panda

OPTFLAGS    = -O 
#INCFLAGS   = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
INCFLAGS    = -I. -I/opt/openmpi-1.4.3-gnu/include
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0 
NVCCFLAGS   = $(CFLAGS) --ptxas-options=-v -arch=sm_20
#LDFLAGS    = -L/N/soft/intel/impi/4.0.0.028/lib64/
LDFLAGS	    = -L/opt/openmpi-1.4.3-gnu/lib/
LIBS        = -lmpi -lmpi_cxx -lpthread -lcutil_x86_64 -lcudart

#  note:
#  with openmpi add -lmpi_cxx
#  gcc usess -fopenmp
#  icc uses -opnemp
#  support wc,dgemm,gemv,c-means,gmm, change the include USERAPI.cu within PandaLib.cu for compiling code

OMPFLAGS    = -fopenmp
CC          = g++
MPICC       = mpicxx
NVCC        = nvcc

NVCCFLAGS  += -Xcompiler -fopenmp
INCFLAGS   += -I/opt/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
INCFLAGS   += -I./include -I./apps/ -I./include/panda -I./
LDFLAGS    += -L/opt/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

APP_CPP_FILES	:= $(wildcard apps/*.cpp)
OS_CPP_FILES 	:= $(wildcard src/oscpp/*.cpp)
PANDA_CPP_FILES := $(wildcard src/panda/*.cpp)
CUDA_CPP_FILES 	:= $(wildcard src/cudacpp/*.cpp)
CUDA_CU_FILES 	:= $(wildcard src/*.cu)

APP_H_FILES	:= $(wildcard apps/*.h)
OS_H_FILES 	:= $(wildcard include/oscpp/*.h)
PANDA_H_FILES 	:= $(wildcard include/panda/*.h)
CUDA_H_FILES 	:= $(wildcard include/cudacpp/*.h)
H_FILES 	:= $(wildcard include/*.h)

APP_OBJ_FILES	:= $(addprefix obj/,$(notdir $(APP_CPP_FILES:.cpp=.o)))
OS_OBJ_FILES 	:= $(addprefix obj/,$(notdir $(OS_CPP_FILES:.cpp=.o)))
PANDA_OBJ_FILES := $(addprefix obj/,$(notdir $(PANDA_CPP_FILES:.cpp=.o)))
CUDA_OBJ_FILES 	:= $(addprefix obj/,$(notdir $(CUDA_CPP_FILES:.cpp=.o)))
CU_OBJ_FILES 	:= $(addprefix cuobj/,$(notdir $(CUDA_CU_FILES:.cu=.o)))

TARGET_OBJ_FILES:=
WC_OBJ_CU_FILES	:= 

panda: panda_cmeans
panda_cmeans: $(APP_OBJ_FILES) $(TARGET_OBJ_FILES) $(OS_OBJ_FILES) $(PANDA_OBJ_FILES) \
		$(CUDA_OBJ_FILES) $(CU_OBJ_FILES)
		g++ $(LIBS) $(LDFLAGS) -o $@ $^

obj/%.o: apps/%.cpp $(APP_H_FILES) 
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/oscpp/%.cpp $(OS_H_FILES)	
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/panda/%.cpp $(PANDA_H_FILES) $(H_FILES)
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/cudacpp/%.cpp $(CUDA_H_FILES)
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: ./%.cpp $(OS_H_FILES) $(PANDA_H_FILES) $(CUDA_H_FILES) $(H_FILES)
	g++ $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

cuobj/%.o: src/%.cu $(CUDA_H_FILES) $(H_FILES)
	nvcc $(LIBS) $(NVCCFLAGS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

clean:
	rm -rf obj/*.o cuobj/*.o panda_cmeans 
