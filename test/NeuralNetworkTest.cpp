#include "NeuralNetwork/NeuralNetwork.hpp"
#include <random>
#include <math.h>

int main() {
    math::Matrix<double> input = math::Matrix<double>(1);
    math::Matrix<double> input2 = math::Matrix<double>(0);
    math::Matrix<double> input3 = math::Matrix<double>({1, 0}, 2, 1);

    math::Matrix<double> desiredOutput = math::Matrix<double>({1, 1, 1});
    math::Matrix<double> desiredOutput2 = math::Matrix<double>({0, 0, 0});
    math::Matrix<double> desiredOutput3 = math::Matrix<double>({{1, 1, 1}, {0, 0, 0}});

    // math::display(input);
    ai::NeuralNetwork<double> net = ai::NeuralNetwork<double>({1, 3});
    std::cout << "Created network." << std::endl;
    std::cout << "Saving weights..." << std::endl;
    net.saveWeights("test/weights");
    std::cout << "Weights saved." << std::endl;
    std::cout << std::endl;
    math::display(net.feedForward(input));
    std::cout << std::endl;
    for (int i = 0; i < 2000; ++i) {
        // std::cout << "================INPUT 1================" << std::endl;
        // net.train(input, desiredOutput, 0.1);
        // std::cout << "================INPUT 2================" << std::endl;
        // net.train(input2, desiredOutput2, 0.1);
        // std::cout << "================INPUT 3================" << std::endl;
        net.train(input3, desiredOutput3, 0.1);
    }
    math::display(net.feedForward(input));
    math::display(net.feedForward(0));
    math::display(net.feedForward(0.7));
    // ai::NeuralNetwork<float> net2;
    // std::cout << "Created network 2." << std::endl;
    // std::cout << "Reading weights..." << std::endl;
    // net2.loadWeights("test/weights");
    // std::cout << "Weights loaded." << std::endl;
    // math::display(input);
}
