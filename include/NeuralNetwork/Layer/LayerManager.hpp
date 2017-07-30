#ifndef LAYER_MANAGER_H
#define LAYER_MANAGER_H
#include <tuple>

namespace ai {
    template <typename Matrix>
    Matrix mse_prime(const Matrix& networkOutput, const Matrix& expectedOutput) {
        return networkOutput - expectedOutput;
    }

    // Taken from https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer?rq=1
    template<int...>
    struct sequence {};

    // Taken from https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer?rq=1
    // How it works: gens<5>: gens<4, 4>: gens<3, 3, 4>: gens<2, 2, 3, 4> : gens<1, 1, 2, 3, 4> : gens<0, 0, 1, 2, 3, 4>.
    // The last type is specialized, creating seq<0, 1, 2, 3, 4>
    template<int N, int... S>
    struct sequenceGenerator : sequenceGenerator<N - 1, N - 1, S...> { };

    template<int... S>
    struct sequenceGenerator<0, S...> {
        typedef sequence<S...> type;
    };

    template <typename Matrix, Matrix (*costDeriv)(const Matrix&, const Matrix&), typename... Layers>
    class LayerManager {
        public:
            LayerManager(Layers&... layers) : layers(layers...) {

            }

            Matrix feedForward(const Matrix& input) {
                return getLayerOutput<sizeof...(Layers)>(input);
            }

            template <int layerNum = 1>
            Matrix getLayerOutput(const Matrix& input) {
                return feedForwardHelper(input, typename sequenceGenerator<layerNum>::type());
            }

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
            // Feed forward unpacker.
            template <int... S>
            inline Matrix feedForwardHelper(const Matrix& input, sequence<S...>) {
                return feedForwardHelper(input, std::get<S>(layers)...);
            }

            // Feed forward base case.
            template <typename BackLayer>
            inline Matrix feedForwardHelper(const Matrix& input, BackLayer& backLayer) {
                return backLayer.feedForward(input);
            }

            // Feed forward recursion.
            template <typename FrontLayer, typename... BackLayers>
            inline Matrix feedForwardHelper(const Matrix& input, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                return feedForwardHelper(frontLayer.feedForward(input), otherLayers...);
            }

            // Backpropagation unpacker.
            template <int... S>
            inline Matrix backpropagateHelper(float learningRate, const Matrix& input, const Matrix& expectedOutput, sequence<S...>) {
                return backpropagateHelper(learningRate, input, expectedOutput, std::get<S>(layers)...);
            }

            // Backpropagation base case.
            template <typename BackLayer>
            inline Matrix backpropagateHelper(float learningRate, const Matrix& input, const Matrix& expectedOutput, BackLayer& backLayer) {
                Matrix weightedOutput = backLayer.getWeightedOutput(input);
                Matrix activationOutput = backLayer.activate(weightedOutput);
                // This will return intermediate deltas for the layer just before.
                return backLayer.template backpropagate<costDeriv>(input, weightedOutput, activationOutput, expectedOutput, learningRate);
            }

            // Backpropagation recursion.
            template <typename FrontLayer, typename... BackLayers>
            inline Matrix backpropagateHelper(float learningRate, const Matrix& input, const Matrix& expectedOutput, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                Matrix layerWeightedOutput = frontLayer.getWeightedOutput(input);
                Matrix layerActivationOutput = frontLayer.activate(layerWeightedOutput);
                // This will give us intermediateDeltas from the next layer.
                Matrix intermediateDeltas = backpropagateHelper(learningRate, layerActivationOutput, expectedOutput, otherLayers...);
                // Use the intermediateDeltas to calculate this layer's deltas, and then pass back other intermediate deltas.
                return frontLayer.backpropagate(input, intermediateDeltas, layerWeightedOutput, learningRate);
            }

            std::tuple<Layers&...> layers;
    };
}
// Define some common types of neural networks here.
template <typename Matrix, typename... Layers>
using NeuralNetwork_MSE = ai::LayerManager<Matrix, ai::mse_prime<Matrix>, Layers...>;


#endif
