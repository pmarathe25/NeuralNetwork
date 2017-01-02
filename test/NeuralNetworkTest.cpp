#include "NeuralNetwork/NeuralNetwork.hpp"
#include <random>
#include <math.h>

int main() {
    math::Matrix<double> input = math::Matrix<double>({1, 2, 3}, 1, 3);
    math::display(input);
    ai::NeuralNetwork<float> net = ai::NeuralNetwork<float>({3, 4000, 4000, 15});
    std::cout << "Created network." << std::endl;
    math::display(net.feedForward(input));
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
    math::display(net2.feedForward(input));
    return 0;
}
