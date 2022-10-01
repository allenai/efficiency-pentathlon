NVCC	:=nvcc
# INC_DIR	:= .
CFLAGS	:=-O3 -std=c++11
LIB_DIR	:=
LIBS	:=-lnvidia-ml

SOURCES := rprof \

all: $(SOURCES)
.PHONY: all

rprof: rprof.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
clean:
	rm -f $(SOURCES)
