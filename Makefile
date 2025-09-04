SM_VER ?= 61
SRC = inverse.cpp
CUDA_EXE = main_cuda.exe
OMP_EXE = main_omp.exe

CUDA_OPT = -O3 -Xcompiler=-fopenmp -std=c++17 --cudart=shared -x cu -I./atlc/include
OMP_OPT = -O3 -fopenmp -std=c++17

.PHONY: cuda omp clean all

all: cuda omp

cuda: $(CUDA_EXE)
	./$(CUDA_EXE)

omp: $(OMP_EXE)
	./$(OMP_EXE)

$(CUDA_EXE): $(SRC)
	nvcc -gencode=arch=compute_$(SM_VER),code=sm_$(SM_VER) $(CUDA_OPT) $< -o $@

$(OMP_EXE): $(SRC)
	g++ $(OMP_OPT) $< -o $@

clean:
	rm -f $(CUDA_EXE) $(OMP_EXE)
