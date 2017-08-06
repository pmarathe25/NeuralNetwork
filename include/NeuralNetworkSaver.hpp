#ifndef NEURAL_NETWORK_SAVER_H
#define NEURAL_NETWORK_SAVER_H
#include "NeuralNetwork.hpp"
#include <tuple>
#include <string>
#include <fstream>

namespace ai {
    class NeuralNetworkSaver {
        public:
            NeuralNetworkSaver() { }

            template <typename Matrix, typename... Layers>
            static void save(NeuralNetwork<Matrix, Layers...>& network, const std::string& filePath) {
                std::ofstream saveFile(filePath);
                if (saveFile.is_open()) {
                    int numLayers = sizeof...(Layers);
                    saveFile.write(reinterpret_cast<char*>(&numLayers), sizeof numLayers);
                    saveUnpacker(saveFile, typename sequenceGenerator<sizeof...(Layers)>::type(), network.getLayers());
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

            template <typename Matrix, typename... Layers>
            static void load(NeuralNetwork<Matrix, Layers...>& network, const std::string& filePath) {
                std::ifstream saveFile(filePath);
                if (saveFile.is_open()) {
                    // Check if it has the correct number of layers.
                    int numLayers = sizeof...(Layers);
                    saveFile.read(reinterpret_cast<char*>(&numLayers), sizeof numLayers);
                    if (numLayers != sizeof...(Layers)) {
                        throw std::invalid_argument("Network depth mismatch during load from file.");
                    }
                    loadUnpacker(saveFile, typename sequenceGenerator<sizeof...(Layers)>::type(), network.getLayers());
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

        private:
            // Weight saving unpacker.
            template <int... S, typename... Layers>
            static inline void saveUnpacker(std::ofstream& saveFile, sequence<S...>, std::tuple<Layers&...> layers) {
                saveRecursive(saveFile, std::get<S>(layers)...);
            }

            // Weight saving base case.
            template <typename BackLayer>
            static inline void saveRecursive(std::ofstream& saveFile, BackLayer& backLayer) {
                backLayer.save(saveFile);
            }

            // Weight saving recursion.
            template <typename FrontLayer, typename... BackLayers>
            static inline void saveRecursive(std::ofstream& saveFile, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                // Save this layer then do the others.
                frontLayer.save(saveFile);
                saveRecursive(saveFile, otherLayers...);
            }

            // Weight loading unpacker.
            template <int... S, typename... Layers>
            static inline void loadUnpacker(std::ifstream& saveFile, sequence<S...>, std::tuple<Layers&...> layers) {
                loadRecursive(saveFile, std::get<S>(layers)...);
            }

            // Weight loading base case.
            template <typename BackLayer>
            static inline void loadRecursive(std::ifstream& saveFile, BackLayer& backLayer) {
                backLayer.load(saveFile);
            }

            // Weight loading recursion.
            template <typename FrontLayer, typename... BackLayers>
            static inline void loadRecursive(std::ifstream& saveFile, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                // Save this layer then do the others.
                frontLayer.load(saveFile);
                loadRecursive(saveFile, otherLayers...);
            }

    };
} /* namespace ai */

#endif
