#ifndef LAYER_H
#define LAYER_H
#include "Matrix.hpp"

namespace ai {
    template <typename T>
    __device__ T sigmoid(T in) {
        return 1 / (1 + exp(-in));
    }

    template <typename T>
    __device__ T analytic(T in) {
        return log(1 + exp(in));
    }

    template <typename T>
    __device__ T relu(T in) {
        return (in < 0) ? 0 : in;
    }

    enum activationFunction {
        SIGMOID = 0,
        ANALYTIC,
        RELU,
    };

    template <typename Matrix>
    class Layer {
    public:
        virtual Matrix feedForward(const Matrix& input) = 0;
    };
}

#endif
