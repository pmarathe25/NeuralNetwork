#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "Layer/Layer.hpp"

namespace ai {
    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    class FullyConnectedLayer : Layer<Matrix> {
    public:
        FullyConnectedLayer(int inputSize, int outputSize);
        FullyConnectedLayer(Matrix weights, Matrix biases);
        // Feeding forward.
        Matrix feedForward(const Matrix& input);
        Matrix getWeightedOutput(const Matrix& input);
        Matrix activate(const Matrix& weightedOutput);
        // Backpropagation for the last layer.
        template <Matrix (*costDeriv)(const Matrix&, const Matrix&)>
        Matrix backpropagate(const Matrix& input, const Matrix& weightedOutput,
            const Matrix& activationOutput, const Matrix& expectedOutput, float learningRate) {
            // z: Weighted outputs of the layer.
            // σ: Activated outputs of the layer.
            // C: The cost function.
            // In order to compute the cost derivative with repect to the weighted inputs, compute (dC / dz) as (dC / dσ) * (dσ / dz)
            Matrix deltas = costDeriv(activationOutput, expectedOutput).hadamard(weightedOutput.template applyFunction<activationDeriv>());
            // Use these deltas and then compute an intermediate quantity for the previous layer.
            return backpropagate(input, deltas, learningRate);
        }
        // Backpropagation for other layers.
        Matrix backpropagate(const Matrix& input, const Matrix& intermediateDeltas, const Matrix& weightedOutput, float learningRate);
    private:
        Matrix weights, biases;
        // Processes deltas and computes a quantity for the previous layer.
        Matrix backpropagate(const Matrix& input, const Matrix& deltas, float learningRate);
        void initializeWeights();
    };
}
// Define some common layers.
template <typename Matrix>
using SigmoidFCL = ai::FullyConnectedLayer<Matrix, ai::sigmoid, ai::sigmoid_prime>;
template <typename Matrix>
using ReLUFCL = ai::FullyConnectedLayer<Matrix, ai::relu, ai::relu_prime>;

#endif
