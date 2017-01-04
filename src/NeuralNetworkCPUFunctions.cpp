#ifndef NEURAL_NETWORK_CPU_FUNCTIONS
#define NEURAL_NETWORK_CPU_FUNCTIONS

namespace ai {
    template <typename T>
    void NeuralNetwork<T>::applySigmoidCPU(math::Matrix<T>& layer) const {
        for (int i = 0; i < layer.size(); ++i) {
            layer[i] = 1 / (1 + exp(layer[i]));
        }
    }

}

#endif
