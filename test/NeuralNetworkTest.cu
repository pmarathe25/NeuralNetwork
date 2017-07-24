#include "NeuralNetwork/NeuralNetwork.hpp"
#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"
#include <random>
#include <math.h>

#include <tuple>

// Taken from https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer?rq=1
template<int...>
struct sequence { };

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
class LayersTest {
    public:
        LayersTest(Layers... layers) : layers(layers...) {

        }

        template <typename T>
        math::Matrix<T> feedForward(const math::Matrix<T>& input) {
           return getLayerOutput<sizeof...(Layers)>(input);
        }

        template <int layerNum = 1, typename T>
        math::Matrix<T> getLayerOutput(const math::Matrix<T>& input) {
            return feedForwardHelper(input, typename sequenceGenerator<layerNum>::type());
        }
    private:

        template <typename T, int... S>
        math::Matrix<T> feedForwardHelper(const math::Matrix<T>& input, sequence<S...>) {
            return feedForwardHelper(input, std::get<S>(layers)...);
        }

        template <typename T, typename BackLayer>
        math::Matrix<T> feedForwardHelper(const math::Matrix<T>& input, BackLayer backLayer) {
            return backLayer.feedForward(input);
        }

        template <typename T, typename FrontLayer, typename... BackLayers>
        math::Matrix<T> feedForwardHelper(const math::Matrix<T>& input, FrontLayer frontLayer, BackLayers... otherLayers) {
            return feedForwardHelper(frontLayer.feedForward(input), otherLayers...);
        }

        std::tuple<Layers...> layers;
};

int main() {
    math::Matrix<double> input({{1, 1, 1}, {0.75, 0.75, 0.75}, {0.5, 0.5, 0.5}, {0.25, 0.25, 0.25}, {0, 0, 0}});
    math::Matrix<double> expectedOutput = input.applyFunction<ai::sigmoid>();

    // input.display();
    ai::NeuralNetwork<double> net({3, 3}, ai::NeuralNetwork<double>::RELU);
    std::cout << "Created network." << std::endl;
    std::cout << "Initial Output" << std::endl;
    net.feedForward(input).display();
    // Train
    for (int i = 0; i < 2000; ++i) {
        net.train(input, expectedOutput, 0.01);
    }
    std::cout << "Actual Output" << std::endl;
    net.feedForward(input).display();

    std::cout << "Expected Output" << std::endl;
    expectedOutput.display();

    // Test Layer functionality
    std::cout << "Testing Fully Connected Layer" << std::endl;
    ai::FullyConnectedLayer<double, ai::relu> testLayer(3, 3);
    ai::FullyConnectedLayer<double, ai::relu> testLayer2(3, 3);
    testLayer.feedForward(input).display();


    LayersTest<ai::FullyConnectedLayer<double, ai::relu>, ai::FullyConnectedLayer<double, ai::relu> > layerTest(testLayer, testLayer2);
    std::cout << "Testing Layer Manager feedForward" << std::endl;
    std::cout << "Expected" << std::endl;
    testLayer2.feedForward(testLayer.feedForward(input)).display();
    std::cout << "Actual" << std::endl;
    layerTest.feedForward(input).display();

    std::cout << "Testing Layer Manager getLayerOutput" << std::endl;
    std::cout << "Expected" << std::endl;
    testLayer.feedForward(input).display();
    std::cout << "Actual" << std::endl;
    layerTest.getLayerOutput(input).display();

    // // Test saving current network
    // std::cout << "Saving weights..." << std::endl;
    // net.saveWeights("test/weights");
    // std::cout << "Weights saved." << std::endl;
    // std::cout << std::endl;
    // // Test loading second network.
    // ai::NeuralNetwork<double> net2;
    // std::cout << "Created network 2." << std::endl;
    // std::cout << "Reading weights..." << std::endl;
    // net2.loadWeights("test/weights");
    // std::cout << "Weights loaded." << std::endl;
}
