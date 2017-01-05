#ifndef NEURAL_NETWORK_CPU_FUNCTIONS
#define NEURAL_NETWORK_CPU_FUNCTIONS

namespace ai {
    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::applySigmoidCPU(const math::Matrix<T>& layer) const {
        math::Matrix<T> out = math::Matrix<T>(layer.numRows(), layer.numColumns());
        for (int i = 0; i < layer.size(); ++i) {
            out[i] = 1 / (1 + exp(-1 * layer[i]));
        }
        return out;
    }

}

#endif
