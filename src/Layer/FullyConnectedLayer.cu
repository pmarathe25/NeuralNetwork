#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"

namespace ai {
    template <typename T, activationFunction aFunc>
    FullyConnectedLayer<T, activationFunction aFunc>::FullyConnectedLayer(int inputSize, int outputSize) {
        this -> weights = math::Matrix<T>(inputSize, outputSize);
        this -> biases = math::Matrix<T>(1, outputSize);
        initializeWeights();
    }

    template <typename T>
    FullyConnectedLayer<T, activationFunction aFunc>::FullyConnectedLayer(math::Matrix<T> weights, math::Matrix<T> biases) {
        this -> weights = weights;
        this -> biases = biases;
    }

    template <typename T>
    math::Matrix<T> FullyConnectedLayer<T, SIGMOID>::feedForward(const math::Matrix<T>& input) {
        return (input * weights).addVector(biases).template applyFunction<sigmoid>();
    }

    template <typename T>
    void FullyConnectedLayer<T, activationFunction aFunc>::initializeWeights() {
        double weightRange = 2 / sqrt(weights.numRows());
        weights.randomizeUniform(-weightRange, weightRange);
        biases.randomizeNormal(0, weightRange);
    }

    template class FullyConnectedLayer<float>;
    template class FullyConnectedLayer<double>;

}
