#ifndef LAYER_H
#define LAYER_H
#include "Math/Matrix.hpp"


namespace ai {
    template <typename T>
    __device__ T sigmoid(T in) {
        return 1 / (1 + exp(-in));
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
        virtual void initializeWeights() = 0;
    };
}

#endif
