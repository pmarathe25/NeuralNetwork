#ifndef NEURAL_NETWORK_OPTIMIZER_H
#define NEURAL_NETWORK_OPTIMIZER_H
#include "NeuralNetwork.hpp"
#include "Minibatch.hpp"
#include <tuple>

namespace ai {
    template <typename Matrix>
    inline Matrix mse_prime(const Matrix& networkOutput, const Matrix& expectedOutput) {
        return networkOutput - expectedOutput;
    }

    template <typename Matrix, Matrix (costDeriv)(const Matrix&, const Matrix&)>
    class NeuralNetworkOptimizer {
        public:
            NeuralNetworkOptimizer() { }

            template <typename... Layers>
            inline Matrix trainMinibatch(NeuralNetwork<Matrix, Layers...>& network, const Minibatch<Matrix>& minibatch, float learningRate = 0.001) {
                return backpropagateUnpacker(learningRate, minibatch.getData(), minibatch.getLabels(), typename sequenceGenerator<sizeof...(Layers)>::type(), network.getLayers());
            }

            // Train over several minibatches for a single epoch.
            template <typename... Layers>
            inline void trainEpoch(NeuralNetwork<Matrix, Layers...>& network, const std::vector<Minibatch<Matrix>>& trainingBatches, float learningRate = 0.001) {
                // Loop over all minibatches.
                for (int j = 0; j < trainingBatches.size(); ++j) {
                    trainMinibatch(network, trainingBatches[j], learningRate);
                }
            }

            // Train for multiple epochs.
            template <int numEpochs = 1, typename... Layers>
            inline void train(NeuralNetwork<Matrix, Layers...>& network, const std::vector<Minibatch<Matrix>>& trainingBatches, float learningRate = 0.001) {
                for (int i = 0; i < numEpochs; ++i) {
                    trainEpoch(network, trainingBatches, learningRate);
                }
            }

        private:
            // Backpropagation unpacker.
            template <int... S, typename... Layers>
            inline Matrix backpropagateUnpacker(float learningRate, const Matrix& input, const Matrix& expectedOutput, sequence<S...>, std::tuple<Layers&...> layers) {
                return backpropagateRecursive(learningRate, input, expectedOutput, std::get<S>(layers)...);
            }

            // Backpropagation base case.
            template <typename BackLayer>
            inline Matrix backpropagateRecursive(float learningRate, const Matrix& input, const Matrix& expectedOutput, BackLayer& backLayer) {
                Matrix layerWeightedOutput = backLayer.getWeightedOutput(input);
                Matrix layerActivationOutput = backLayer.activate(layerWeightedOutput);
                // Compute cost derivative.
                Matrix intermediateDeltas = costDeriv(layerActivationOutput, expectedOutput);
                // Now compute deltas
                Matrix deltas = backLayer.computeDeltas(intermediateDeltas, layerWeightedOutput);
                // This will return intermediate deltas for the layer before this one.
                Matrix previousIntermediateDeltas = backLayer.backpropagate(deltas);
                // Adjust weights + biases AFTER computing deltas for the previous layer, so our new weights don't affect the current training iteration.
                backLayer.sgd(input, deltas, learningRate);
                return previousIntermediateDeltas;
            }

            // Backpropagation recursion.
            template <typename FrontLayer, typename... BackLayers>
            inline Matrix backpropagateRecursive(float learningRate, const Matrix& input, const Matrix& expectedOutput, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                Matrix layerWeightedOutput = frontLayer.getWeightedOutput(input);
                Matrix layerActivationOutput = frontLayer.activate(layerWeightedOutput);
                // This will give us intermediateDeltas from the next layer.
                Matrix intermediateDeltas = backpropagateRecursive(learningRate, layerActivationOutput, expectedOutput, otherLayers...);
                // Use the intermediateDeltas to calculate this layer's deltas.
                Matrix deltas = frontLayer.computeDeltas(intermediateDeltas, layerWeightedOutput);
                // Now compute intermediate deltas for the layer before this one.
                Matrix previousIntermediateDeltas = frontLayer.backpropagate(deltas);
                // Adjust weights + biases AFTER computing deltas for the previous layer, so our new weights don't affect the current training iteration.
                frontLayer.sgd(input, deltas, learningRate);
                return previousIntermediateDeltas;
            }
    };
}


#endif
