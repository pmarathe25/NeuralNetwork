#ifndef NEURAL_NETWORK_OPTIMIZER_H
#define NEURAL_NETWORK_OPTIMIZER_H
#include "NeuralNetwork.hpp"

namespace ai {
    template <typename Matrix>
    Matrix mse_prime(const Matrix& networkOutput, const Matrix& expectedOutput) {
        return networkOutput - expectedOutput;
    }

    template <typename Matrix, Matrix (*costDeriv)(const Matrix&, const Matrix&), typename... Layers>
    class NeuralNetworkOptimizer {
        public:
            NeuralNetworkOptimizer(NeuralNetwork<Matrix, Layers...>& target) : layers(target.getLayers()) { }

            Matrix backpropagate(const Matrix& input, const Matrix& expectedOutput, float learningRate = 0.001) {
                return backpropagateHelper(learningRate, input, expectedOutput, typename sequenceGenerator<sizeof...(Layers)>::type());
            }

            template <int traningIters = 2000>
            void train(const Matrix& input, const Matrix& expectedOutput, float learningRate = 0.001) {
                // Ask the compiler to unroll this, since we know trainingIters at compile time.
                #pragma unroll
                for (int i = traningIters; i > 0; --i) {
                    backpropagate(input, expectedOutput, learningRate);
                }
            }
        private:
            // Backpropagation unpacker.
            template <int... S>
            inline Matrix backpropagateHelper(float learningRate, const Matrix& input, const Matrix& expectedOutput, sequence<S...>) {
                return backpropagateHelper(learningRate, input, expectedOutput, std::get<S>(layers)...);
            }

            // Backpropagation base case.
            template <typename BackLayer>
            inline Matrix backpropagateHelper(float learningRate, const Matrix& input, const Matrix& expectedOutput, BackLayer& backLayer) {
                Matrix layerWeightedOutput = backLayer.getWeightedOutput(input);
                Matrix layerActivationOutput = backLayer.activate(layerWeightedOutput);
                // Compute cost derivative.
                Matrix intermediateDeltas = costDeriv(layerActivationOutput, expectedOutput);
                // Now compute deltas
                Matrix deltas = backLayer.computeDeltas(input, intermediateDeltas, layerWeightedOutput, learningRate);
                // This will return intermediate deltas for the layer before this one.
                Matrix previousIntermediateDeltas = backLayer.backpropagate(input, deltas, learningRate);
                return previousIntermediateDeltas;
            }

            // Backpropagation recursion.
            template <typename FrontLayer, typename... BackLayers>
            inline Matrix backpropagateHelper(float learningRate, const Matrix& input, const Matrix& expectedOutput, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                Matrix layerWeightedOutput = frontLayer.getWeightedOutput(input);
                Matrix layerActivationOutput = frontLayer.activate(layerWeightedOutput);
                // This will give us intermediateDeltas from the next layer.
                Matrix intermediateDeltas = backpropagateHelper(learningRate, layerActivationOutput, expectedOutput, otherLayers...);
                // Use the intermediateDeltas to calculate this layer's deltas.
                Matrix deltas = frontLayer.computeDeltas(input, intermediateDeltas, layerWeightedOutput, learningRate);
                // Now compute intermediate deltas for the layer before this one.
                Matrix previousIntermediateDeltas = frontLayer.backpropagate(input, deltas, learningRate);
                return previousIntermediateDeltas;
            }

            std::tuple<Layers&...> layers;
    };
}


#endif
