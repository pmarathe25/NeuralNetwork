#include "NeuralNetwork/NeuralNetwork.hpp"
#include <random>
#include <math.h>

int main() {
    std::vector<double> input = std::vector<double>({1, 2, 3});
    math::display(input);
    ai::NeuralNetwork<double> net = ai::NeuralNetwork<double>({3, 400, 400, 15});
    std::cout << "Created network." << std::endl;
    for (int i = 0; i < 100; ++i) {
        math::display(net.feedForward(input));
    }
    std::cout << "Saving weights..." << std::endl;
    net.saveWeights("test/weights");
    std::cout << "Weights saved." << std::endl;
    std::cout << std::endl;
    ai::NeuralNetwork<float> net2;
    std::cout << "Created network 2." << std::endl;
    std::cout << "Reading weights..." << std::endl;
    net2.loadWeights("test/weights");
    std::cout << "Weights loaded." << std::endl;
    math::display(input);
    std::vector<float> input2 = std::vector<float>({1, 2, 3});
    math::display(net2.feedForward(input2));
    return 0;
}
