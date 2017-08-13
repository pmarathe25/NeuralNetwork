#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <tuple>
#include <string>

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

    template <typename Matrix, typename... Layers>
    class NeuralNetwork {
        public:
            NeuralNetwork(const std::string& name, Layers&... layers) : layers(layers...), name(name) { }

            Matrix feedForward(const Matrix& input) {
                return getLayerOutput<sizeof...(Layers)>(input);
            }

            template <int layerNum = 1>
            Matrix getLayerOutput(const Matrix& input) {
                return feedForwardUnpacker(input, typename sequenceGenerator<layerNum>::type());
            }

            int getDepth() {
                return sizeof...(Layers);
            }

            const std::string& getName() const {
                return name;
            }

            // Return a tuple of const references to layers.
            const std::tuple<Layers&...> getLayers() const {
                return layers;
            }

            // Return a tuple of non-const references to layers.
            std::tuple<Layers&...> getLayers() {
                return layers;
            }

        private:
            // Feed forward unpacker.
            template <int... S>
            inline Matrix feedForwardUnpacker(const Matrix& input, sequence<S...>) {
                return feedForwardRecursive(input, std::get<S>(layers)...);
            }

            // Feed forward base case.
            template <typename BackLayer>
            inline Matrix feedForwardRecursive(const Matrix& input, BackLayer& backLayer) {
                return backLayer.feedForward(input);
            }

            // Feed forward recursion.
            template <typename FrontLayer, typename... BackLayers>
            inline Matrix feedForwardRecursive(const Matrix& input, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                return feedForwardRecursive(frontLayer.feedForward(input), otherLayers...);
            }

            std::tuple<Layers&...> layers;
            const std::string name;
    };
}

#endif
