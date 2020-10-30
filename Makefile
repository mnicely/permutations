NVCC	:=nvcc --cudart=static -ccbin g++
CFLAGS	:=-O3 -std=c++14 -lineinfo
ARCHES	:=-gencode arch=compute_70,code=\"compute_70,sm_70\" -gencode arch=compute_75,code=\"compute_75,sm_75\" -gencode arch=compute_80,code=\"compute_80,sm_80\"
INC_DIR	:=
LIB_DIR	:=
LIBS	:=

SOURCES := permutations

all: $(SOURCES)
.PHONY: all

permutations: permutations.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

clean:
	@echo 'Cleaning up...'
	@echo 'rm -rf $(SOURCES)'
	@rm -rf $(SOURCES) 