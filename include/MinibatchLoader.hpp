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
                        trainingSet.emplace_back(filename);
                    }
                }
                return trainingSet;
            }

            // TODO: Shuffle the training set.
            void shuffle() {

            }

            // TODO: Partition the data set according to ratios
            void partitionTrainingSet(float training = 100, float validation = 0, float test = 0) {
                int total = std::ceil(training + validation + test);
                if (total == 100) {

                } else {
                    throw std::invalid_argument("Training Set sizes must add to 100%.")
                }
            }

            const std::vector<Minibatch<Matrix>>& getTrainingData() const {
                return trainingSet;
            }
        private:
            std::vector<Minibatch<Matrix>> trainingSet, validationSet, testSet;
    };
} /* namespace ai */

#endif
