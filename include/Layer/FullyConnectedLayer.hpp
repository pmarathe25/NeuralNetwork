#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "Layer/Layer.hpp"

namespace ai {
    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    class FullyConnectedLayer : Layer<Matrix> {
    public:
        // Default weight initialization if needed.
        FullyConnectedLayer(int inputSize, int outputSize);
        // Custom weight initialization is much preferred.
        FullyConnectedLayer(Matrix weights, Matrix biases);
        // Feeding forward.
        Matrix feedForward(const Matrix& input) const;
        Matrix getWeightedOutput(const Matrix& input) const;
        Matrix activate(const Matrix& weightedOutput) const;
        // Backpropagation for other layers.
        Matrix computeDeltas(const Matrix& input, const Matrix& intermediateDeltas, const Matrix& weightedOutput, float learningRate);
        Matrix backpropagate(const Matrix& input, const Matrix& deltas, float learningRate);
        Matrix weights, biases;
    private:
        // Processes deltas and computes a quantity for the previous layer.
        void initializeWeights();
    };
} /* namespace ai */
// Define some common layers.
template <typename Matrix>
using SigmoidFCL = ai::FullyConnectedLayer<Matrix, ai::sigmoid, ai::sigmoid_prime>;
template <typename Matrix>
using ReLUFCL = ai::FullyConnectedLayer<Matrix, ai::relu, ai::relu_prime>;
template <typename Matrix>
using LeakyReLUFCL = ai::FullyConnectedLayer<Matrix, ai::leakyRelu, ai::leakyRelu_prime>;

#endif
