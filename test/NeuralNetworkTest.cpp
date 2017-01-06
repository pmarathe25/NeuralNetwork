#include "NeuralNetwork/NeuralNetwork.hpp"
#include <random>
#include <math.h>

int main() {
    math::Matrix<double> input = math::Matrix<double>({1, 0}, 2, 1);
    math::Matrix<double> desiredOutput = math::Matrix<double>({{1, 1, 1}, {0, 0, 0}});
    // math::display(input);
    ai::NeuralNetwork<double> net = ai::NeuralNetwork<double>({1, 1, 3});
    std::cout << "Created network." << std::endl;
    std::cout << "Saving weights..." << std::endl;
    net.saveWeights("test/weights");
    std::cout << "Weights saved." << std::endl;
    std::cout << std::endl;
    math::display(net.feedForward(input));
    // math::display(net.feedForward(input2));
    std::cout << std::endl;
    for (int i = 0; i < 2000; ++i) {
        net.train(input, desiredOutput, 0.1);
        // net.train(input2, desiredOutput2, 0.1);
    }
    math::display(net.feedForward(input));
    math::display(net.feedForward(0.2));
    // math::display(net.feedForward(0.7));
    // ai::NeuralNetwork<float> net2;
    // std::cout << "Created network 2." << std::endl;
    // std::cout << "Reading weights..." << std::endl;
    // net2.loadWeights("test/weights");
    // std::cout << "Weights loaded." << std::endl;
    // math::display(input);
}
