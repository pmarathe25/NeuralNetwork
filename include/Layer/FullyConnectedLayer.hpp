#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "Layer/Layer.hpp"

namespace ai {

    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    class FullyConnectedLayer : Layer<Matrix> {
    public:
        FullyConnectedLayer(int inputSize, int outputSize) {
            this -> weights = Matrix(inputSize, outputSize);
            this -> biases = Matrix(1, outputSize);
            initializeWeights();
        }

        FullyConnectedLayer(Matrix weights, Matrix biases) {
            this -> weights = weights;
            this -> biases = biases;
        }

        inline Matrix feedForward(const Matrix& input) {
            return activate(getWeightedOutput(input));
        }

        inline Matrix getWeightedOutput(const Matrix& input) {
            return (input * weights).addVector(biases);
        }

        inline Matrix activate(const Matrix& weightedOutput) {
            return weightedOutput.template applyFunction<activationFunc>();
        }

        // Backpropagation for the last layer.
        template <Matrix (*costDeriv)(const Matrix&, const Matrix&)>
        inline Matrix backpropagate(const Matrix& input, const Matrix& weightedOutput, const Matrix& activationOutput, const Matrix& expectedOutput, float learningRate) {
            // z: Weighted outputs of the layer.
            // σ: Activated outputs of the layer.
            // C: The cost function.
            // In order to compute the cost derivative with repect to the weighted inputs, compute (dC / dz) as (dC / dσ) * (dσ / dz)
            Matrix deltas = costDeriv(activationOutput, expectedOutput).hadamard(weightedOutput.template applyFunction<activationDeriv>());
            // Use these deltas and then compute an intermediate quantity for the previous layer.
            return backpropagate(input, deltas, learningRate);
        }

        // Backpropagation for other layers.
        inline Matrix backpropagate(const Matrix& input, const Matrix& intermediateDeltas, const Matrix& weightedOutput, float learningRate) {
            // Compute this layer's deltas
            Matrix deltas = intermediateDeltas.hadamard(weightedOutput.template applyFunction<activationDeriv>());
            // Use these deltas and then compute an intermediate quantity for the previous layer.
            return backpropagate(input, deltas, learningRate);
        }

        void initializeWeights() {
            double weightRange = 2 / sqrt(weights.numRows());
            if (activationFunc == static_cast<float (*)(float)>(relu<float>)) {
                weights = Matrix::randomUniformLike(weights, 0, weightRange);
                biases = Matrix::randomNormalLike(biases, 0, weightRange).template applyFunction<abs>();
            } else {
                weights = Matrix::randomUniformLike(weights, -weightRange, weightRange);
                biases = Matrix::randomNormalLike(biases, 0, weightRange);
            }
        }
    private:
        Matrix weights, biases;
        // Processes deltas and computes a quantity for the previous layer.
        inline Matrix backpropagate(const Matrix& input, const Matrix& deltas, float learningRate) {
            // For the previous layer.
            Matrix intermediateDeltas = deltas * weights.transpose();
            // Modify this layer's weights.
            weights -= input.transpose() * deltas * learningRate;
            // Return an intermediate quantity for the previous layer.
            return intermediateDeltas;
        }
    };
}
// Define some common layers.
template <typename Matrix>
using SigmoidFCL = ai::FullyConnectedLayer<Matrix, ai::sigmoid, ai::sigmoid_prime>;
template <typename Matrix>
using ReLUFCL = ai::FullyConnectedLayer<Matrix, ai::relu, ai::relu_prime>;

#endif
