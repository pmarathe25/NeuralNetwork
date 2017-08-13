#ifndef NEURAL_NETWORK_OPTIMIZER_H
#define NEURAL_NETWORK_OPTIMIZER_H
#include "NeuralNetwork.hpp"
#include "Minibatch.hpp"
#include <tuple>
#include <dirent.h>

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
            Matrix backpropagate(NeuralNetwork<Matrix, Layers...>& network, const Matrix& input, const Matrix& expectedOutput, float learningRate = 0.001) {
                return backpropagateUnpacker(learningRate, input, expectedOutput, typename sequenceGenerator<sizeof...(Layers)>::type(), network.getLayers());
            }

            std::vector<Minibatch<Matrix>> loadMinibatches(const std::string& minibatchFolder) {
                std::vector<Minibatch<Matrix>> trainingData;
                // Load all minibatches from the folder.
                DIR* dir = opendir(minibatchFolder.c_str());
                for (dirent* d = readdir(dir); d != NULL; d = readdir(dir)) {
                    std::string filename = minibatchFolder + "/" + d -> d_name;
                    if (Minibatch<Matrix>::isMinibatchFile(filename)) {
                        trainingData.emplace_back(filename);
                    }
                }
                return trainingData;
            }

            template <int numEpochs = 1, typename... Layers>
            void train(NeuralNetwork<Matrix, Layers...>& network, const std::vector<Minibatch<Matrix>>& trainingBatches, float learningRate = 0.001) {
                for (int i = 0; i < numEpochs; ++i) {
                    // Loop over all minibatches.
                    for (int j = 0; j < trainingBatches.size(); ++j) {
                        trainMinibatch(network, trainingBatches[j], learningRate);
                    }
                    std::cout << "Finished Epoch " << i << '\n';
                }
            }

            template <int numEpochs = 1, typename... Layers>
            void train(NeuralNetwork<Matrix, Layers...>& network, const std::string& minibatchFolder, float learningRate = 0.001) {
                std::vector<Minibatch<Matrix>> trainingData = loadMinibatches(minibatchFolder);
                train<numEpochs>(network, trainingData, learningRate);
            }

            template <int traningIters = 1, typename... Layers>
            void trainMinibatch(NeuralNetwork<Matrix, Layers...>& network, const Minibatch<Matrix>& minibatch, float learningRate = 0.001) {
                for (int i = traningIters; i > 0; --i) {
                    backpropagate(network, minibatch.getData(), minibatch.getLabels(), learningRate);
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
