#ifndef LAYER_MANAGER_H
#define LAYER_MANAGER_H
#include <tuple>

namespace ai {
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

    template <typename... Layers>
    class LayerManager {
        public:
            LayerManager(Layers... layers) : layers(layers...) {

            }

            template <typename Matrix>
            Matrix feedForward(const Matrix& input) {
                return getLayerOutput<sizeof...(Layers)>(input);
            }

            template <typename Matrix>
            Matrix backpropagate(const Matrix& input, const Matrix& expectedOutput) {
                return backpropagateHelper(input, expectedOutput, typename sequenceGenerator<sizeof...(Layers)>::type());
            }

            template <int layerNum = 1, typename Matrix>
            Matrix getLayerOutput(const Matrix& input) {
                return feedForwardHelper(input, typename sequenceGenerator<layerNum>::type());
            }
        private:
            // Feed forward unpacker.
            template <typename Matrix, int... S>
            Matrix feedForwardHelper(const Matrix& input, sequence<S...>) {
                return feedForwardHelper(input, std::get<S>(layers)...);
            }

            // Feed forward base case.
            template <typename Matrix, typename BackLayer>
            Matrix feedForwardHelper(const Matrix& input, BackLayer backLayer) {
                return backLayer.feedForward(input);
            }

            // Feed forward recursion.
            template <typename Matrix, typename FrontLayer, typename... BackLayers>
            Matrix feedForwardHelper(const Matrix& input, FrontLayer frontLayer, BackLayers... otherLayers) {
                return feedForwardHelper(frontLayer.feedForward(input), otherLayers...);
            }

            // Backpropagation unpacker.
            template <typename Matrix, int... S>
            Matrix backpropagateHelper(const Matrix& input, const Matrix& expectedOutput, sequence<S...>) {
                return backpropagateHelper(input, expectedOutput, std::get<S>(layers)...);
            }

            // Backpropagation base case.
            template <typename Matrix, typename BackLayer>
            Matrix backpropagateHelper(const Matrix& input, const Matrix& expectedOutput, BackLayer backLayer) {
                Matrix weightedOutput = backLayer.getWeightedOutput(input);
                Matrix activationOutput = backLayer.activate(weightedOutput);
                return backLayer.backpropagate(activationOutput, expectedOutput);
            }

            // Backpropagation recursion.
            template <typename Matrix, typename FrontLayer, typename... BackLayers>
            Matrix backpropagateHelper(const Matrix& input, const Matrix& expectedOutput, FrontLayer frontLayer, BackLayers... otherLayers) {
                Matrix layerWeightedOutput = frontLayer.getWeightedOutput(input);
                Matrix layerActivationOutput = frontLayer.activate(layerWeightedOutput);
                Matrix layerDeltas = backpropagateHelper(layerActivationOutput, expectedOutput, otherLayers...);
                return frontLayer.backpropagate(layerDeltas);
            }

            std::tuple<Layers...> layers;
        };
}


#endif
