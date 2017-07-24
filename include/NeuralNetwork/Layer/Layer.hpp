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

    template <typename T>
    class Layer {
    public:
        virtual math::Matrix<T> feedForward(const math::Matrix<T>& input) = 0;
    };
}

#endif
