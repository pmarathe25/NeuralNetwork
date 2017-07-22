#ifndef NEURAL_NETWORK_CUDA_FUNCTIONS
#define NEURAL_NETWORK_CUDA_FUNCTIONS

namespace ai {
    template <typename T>
    __global__ void activationFunctionSigmoid(const T* input, int size, T* output) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            output[index] = 1 / (1 + exp(-input[index]));
        }
    }

    template <typename T>
    __global__ void activationFunctionRELUDerivative(const T* input, int size, T* output) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            output[index] = input[index] > 0;
        }
    }

    template <typename T>
    __global__ void computeFeedForward(const T* input, const T* weight, const T* bias, int numRowsActivatedOutput,
        int numColsActivatedOutput, int numColsWeight, int inputSize, int weightSize, T* output, T* activatedOutput, int aFunc) {
        __shared__ T tileActivated[BLOCK_DIM][BLOCK_DIM + 1];
        __shared__ T tileWeight[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        bool cValid = row < numRowsActivatedOutput && col < numColsWeight;
        T outValue = T();
        // Iterate over the sub-matrices of A and B.
        int maxIterations = numColsActivatedOutput + BLOCK_DIM - 1;
        for (int i = 0; i < maxIterations; i += BLOCK_DIM) {
            // Compute indices.
            int indexActivatedOutput = row * numColsActivatedOutput + (i + threadIdx.y);
            int indexWeight = (i + threadIdx.x) * numColsWeight + col;
            // Load sub-matrix A.
            tileActivated[threadIdx.x][threadIdx.y] = (indexActivatedOutput < inputSize) ? input[indexActivatedOutput] : 0;
            // Load sub-matrix B.
            tileWeight[threadIdx.x][threadIdx.y] = (indexWeight < weightSize) ? weight[indexWeight] : 0;
            // Synchronize.
            __syncthreads();
            // Compute dot product only if the point is within the C matrix.
            if (cValid) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                   outValue += tileActivated[threadIdx.x][j] * tileWeight[j][threadIdx.y];
                }
            }
            // Synchronize.
            __syncthreads();
        }
        // Write to output.
        if (cValid) {
            outValue = outValue + bias[col];
            output[row * numColsWeight + col] = outValue;
            switch (aFunc) {
               case 0:
                   activatedOutput[row * numColsWeight + col] = 1 / (1 + exp(-outValue));
                   break;
               case 1:
                   activatedOutput[row * numColsWeight + col] = log(1 + exp(outValue));
                   break;
               case 2:
                   activatedOutput[row * numColsWeight + col] = (outValue < 0) ? 0 : outValue;
                   break;
            }
        }
    }
}

#endif
