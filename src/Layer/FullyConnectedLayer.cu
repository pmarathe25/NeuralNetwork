#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"

namespace ai {
    template <typename T>
    FullyConnectedLayer<T>::FullyConnectedLayer(int inputSize, int outputSize) {
        this -> weights = math::Matrix<T>(inputSize, outputSize);
        this -> biases = math::Matrix<T>(1, outputSize);
        initializeWeights();
    }

    template <typename T>
    FullyConnectedLayer<T>::FullyConnectedLayer(math::Matrix<T> weights, math::Matrix<T> biases) {
        this -> weights = weights;
        this -> biases = biases;
    }

    template <typename T>
    math::Matrix<T> FullyConnectedLayer<T>::feedForward(const math::Matrix<T>& input) {
        return (input * weights).addVector(biases).template applyFunction<sigmoid>();
    }

    template <typename T>
    void FullyConnectedLayer<T>::initializeWeights() {
        double weightRange = 2 / sqrt(weights.numRows());
        weights.randomizeUniform(-weightRange, weightRange);
        biases.randomizeNormal(0, weightRange);
    }

    template class FullyConnectedLayer<float>;
    template class FullyConnectedLayer<double>;

}
