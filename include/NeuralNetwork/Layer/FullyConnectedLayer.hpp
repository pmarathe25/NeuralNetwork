#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "NeuralNetwork/Layer/Layer.hpp"

namespace ai {
    template <typename T, activationFunction aFunc>
    class FullyConnectedLayer : Layer<T> {
    public:
        FullyConnectedLayer(int inputSize, int outputSize);
        FullyConnectedLayer(math::Matrix<T> weights, math::Matrix<T> biases);
        math::Matrix<T> feedForward(const math::Matrix<T>& input);
        void initializeWeights();
    private:
        math::Matrix<T> weights, biases;
    };
}

#endif
