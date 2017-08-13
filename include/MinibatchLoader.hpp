#ifndef MINIBATCH_LOADER_H
#define MINIBATCH_LOADER_H
#include "Minibatch.hpp"
#include <string>
#include <vector>
#include <dirent.h>

namespace ai {
    template <typename Matrix>
    class MinibatchLoader {
        public:
            MinibatchLoader() { }

            MinibatchLoader(const std::string& minibatchFolder) {
                loadMinibatches(minibatchFolder);
            }

            const std::vector<Minibatch<Matrix>>& loadMinibatches(const std::string& minibatchFolder) {
                // Load all minibatches from the folder.
                DIR* dir = opendir(minibatchFolder.c_str());
                for (dirent* d = readdir(dir); d != NULL; d = readdir(dir)) {
                    std::string filename = minibatchFolder + "/" + d -> d_name;
                    if (Minibatch<Matrix>::isMinibatchFile(filename)) {
                        trainingData.emplace_back(filename);
                    }
                }
                return trainingData;
            }

            const std::vector<Minibatch<Matrix>>& getTrainingData() const {
                return trainingData;
            }
        private:
            std::vector<Minibatch<Matrix>> trainingData;
    };
} /* namespace ai */

#endif
