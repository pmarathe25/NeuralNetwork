#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "NeuralNetwork/Layer/Layer.hpp"

#include <iostream>

namespace ai {
    template <typename T, T (*activationFunc)(T)>
    class FullyConnectedLayer : Layer<T> {
    public:
        FullyConnectedLayer(int inputSize, int outputSize) {
            this -> weights = math::Matrix<T>(inputSize, outputSize);
            this -> biases = math::Matrix<T>(1, outputSize);
            initializeWeights();
        }

        FullyConnectedLayer(math::Matrix<T> weights, math::Matrix<T> biases) {
            this -> weights = weights;
            this -> biases = biases;
        }

        math::Matrix<T> feedForward(const math::Matrix<T>& input) {
            return (input * weights).addVector(biases).template applyFunction<activationFunc>();
        }

        void initializeWeights() {
            double weightRange = 2 / sqrt(weights.numRows());
            if (activationFunc == static_cast<T (*)(T)>(relu<T>)) {
                weights = randomUniformLike(weights, 0, weightRange);
                biases = randomNormalLike(biases, 0, weightRange).template applyFunction<abs>();
            } else {
                weights = randomUniformLike(weights, -weightRange, weightRange);
                biases = randomNormalLike(biases, 0, weightRange);
            }
        }
    private:
        math::Matrix<T> weights, biases;
    };
}

#endif
