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

        Matrix feedForward(const Matrix& input) const {
            return activate(getWeightedOutput(input));
        }

        Matrix getWeightedOutput(const Matrix& input) const {
            return (input * weights).addVector(biases);
        }

        Matrix activate(const Matrix& weightedOutput) const {
            return weightedOutput.template applyFunction<activationFunc>();
        }

        Matrix computeDeltas(const Matrix& intermediateDeltas, const Matrix& weightedOutput) const {
            // Compute this layer's deltas
            return intermediateDeltas.hadamard(weightedOutput.template applyFunction<activationDeriv>());
        }

        Matrix backpropagate(const Matrix& deltas) {
            return deltas * weights.transpose();
        }

        // Train!
        void sgd(const Matrix& input, const Matrix& deltas, float learningRate) {
            weights -= input.transpose() * deltas * learningRate / (float) deltas.numRows();
            biases -= deltas.rowMean() * learningRate;
        }

        const Matrix& getWeights() const {
            return weights;
        }

        const Matrix& getBiases() const {
            return biases;
        }

    private:
        Matrix weights, biases;
        // Processes deltas and computes a quantity for the previous layer.
        void initializeWeights() {
            double weightRange = 2 / sqrt(weights.numRows());
            weights = Matrix::randomUniformLike(weights, -weightRange, weightRange);
            biases = Matrix::randomNormalLike(biases, 0, weightRange);
        }
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
