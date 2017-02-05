#ifndef NEURAL_NETWORK_CUDA_FUNCTIONS
#define NEURAL_NETWORK_CUDA_FUNCTIONS

namespace ai {
    template <typename T>
    __global__ void activationFunctionSigmoid(T* mat, int size, T* output) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            output[index] = 1 / (1 + exp(-mat[index]));
        }
    }
}

#endif
