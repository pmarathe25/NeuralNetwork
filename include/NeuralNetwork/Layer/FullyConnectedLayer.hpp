#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "NeuralNetwork/Layer/Layer.hpp"

namespace ai {
    template <typename Matrix, float (*activationFunc)(float)>
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

        inline Matrix backpropagate(const Matrix& networkOutput, const Matrix& expectedOutput) {

        }

        inline Matrix backpropagate(const Matrix& layerDeltas) {


        }

        void initializeWeights() {
            double weightRange = 2 / sqrt(weights.numRows());
            if (activationFunc == static_cast<float (*)(float)>(relu<float>)) {
                weights = randomUniformLike(weights, 0, weightRange);
                biases = randomNormalLike(biases, 0, weightRange).template applyFunction<abs>();
            } else {
                weights = randomUniformLike(weights, -weightRange, weightRange);
                biases = randomNormalLike(biases, 0, weightRange);
            }
        }
    private:
        Matrix weights, biases;
    };

    // Define some common layers.
    typedef FullyConnectedLayer<math::Matrix<float>, ai::sigmoid> SigmoidFCL;
    typedef FullyConnectedLayer<math::Matrix<float>, ai::relu> ReLUFCL;
}

#endif
