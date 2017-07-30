#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "NeuralNetwork/Layer/Layer.hpp"
#include "Matrix.hpp"

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

        template <Matrix (*costDeriv)(const Matrix&, const Matrix&)>
        inline Matrix computeBackDeltas(const Matrix& weightedOutput, const Matrix& activationOutput, const Matrix& expectedOutput) {
            // z: weighted outputs of the layer.
            // σ: activated outputs of the layer.
            // C: the cost function.
            // In order to compute the cost derivative with repect to the weighted inputs,
            // compute (dC / dz) as (dC / dσ) * (dσ / dz)
            return costDeriv(activationOutput, expectedOutput).hadamard(weightedOutput.template applyFunction<activationDeriv>());
        }

        inline Matrix backpropagate(const Matrix& layerDeltas) {


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
    };

    // Define some common layers.
    template <float (*activationFunc)(float), float (*activationDeriv)(float)>
    using FCL = FullyConnectedLayer<Matrix_F, activationFunc, activationDeriv>;
    // Now with some common activation functions.
    typedef FCL<ai::sigmoid, ai::sigmoid_prime> SigmoidFCL;
    typedef FCL<ai::relu, ai::relu_prime> ReLUFCL;
}

#endif
