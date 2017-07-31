#include "Matrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"

namespace ai{
    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::FullyConnectedLayer(int inputSize, int outputSize) {
        this -> weights = Matrix(inputSize, outputSize);
        this -> biases = Matrix(1, outputSize);
        initializeWeights();
    }

    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::FullyConnectedLayer(Matrix weights, Matrix biases) {
        this -> weights = weights;
        this -> biases = biases;
    }

    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    Matrix FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::feedForward(const Matrix& input) const {
        return activate(getWeightedOutput(input));
    }

    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    Matrix FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::getWeightedOutput(const Matrix& input) const {
        return (input * weights).addVector(biases);
    }

    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    Matrix FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::activate(const Matrix& weightedOutput) const {
        return weightedOutput.template applyFunction<activationFunc>();
    }

    // Backpropagation for other layers.
    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    Matrix FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::backpropagate(const Matrix& input, const Matrix& intermediateDeltas, const Matrix& weightedOutput, float learningRate) {
        // Compute this layer's deltas
        Matrix deltas = intermediateDeltas.hadamard(weightedOutput.template applyFunction<activationDeriv>());
        // Use these deltas and then compute an intermediate quantity for the previous layer.
        return backpropagate(input, deltas, learningRate);
    }

    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    void FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::initializeWeights() {
        double weightRange = 2 / sqrt(weights.numRows());
        weights = Matrix::randomUniformLike(weights, -weightRange, weightRange);
        biases = Matrix::randomNormalLike(biases, 0, weightRange);
    }

    // Processes deltas and computes a quantity for the previous layer.
    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    Matrix FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::backpropagate(const Matrix& input, const Matrix& deltas, float learningRate) {
        // For the previous layer.
        Matrix intermediateDeltas = deltas * weights.transpose();
        // Modify this layer's weights.
        weights -= input.transpose() * deltas * learningRate / deltas.numRows();
        biases -= deltas.rowMean() * learningRate;
        // Return an intermediate quantity for the previous layer.
        return intermediateDeltas;
    }

    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    const Matrix& FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::getWeights() const {
        return weights;
    }

    template <typename Matrix, float (*activationFunc)(float), float (*activationDeriv)(float)>
    const Matrix& FullyConnectedLayer<Matrix, activationFunc, activationDeriv>::getBiases() const {
        return biases;
    }


    template class FullyConnectedLayer<Matrix_F, ai::sigmoid, ai::sigmoid_prime>;
    template class FullyConnectedLayer<Matrix_F, ai::relu, ai::relu_prime>;

} /* namespace ai */
