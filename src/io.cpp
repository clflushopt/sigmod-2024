
#include "io.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace knn {

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>> &knns,
             const std::string &path) {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  const int K = 100;
  const uint32_t N = knns.size();
  assert(knns.front().size() == K);
  for (unsigned i = 0; i < N; ++i) {
    auto const &knn = knns[i];
    ofs.write(reinterpret_cast<char const *>(knn.data()), K * sizeof(uint32_t));
  }
  ofs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path, const int num_dimensions,
             std::vector<std::vector<float>> &data) {
  std::cout << "Reading Data: " << file_path << '\n';
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N = 0;  // num of points
  ifs.read(reinterpret_cast<char *>(&N), sizeof(uint32_t));
  data.resize(N);
  std::cout << "# of points: " << N << '\n';
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read(reinterpret_cast<char *>(buff.data()),
                  num_dimensions * sizeof(float))) {
    std::vector<float> row(num_dimensions);
    for (int dim = 0; dim < num_dimensions; dim++) {
      row[dim] = static_cast<float>(buff[dim]);
    }
    data[counter++] = std::move(row);
  }
  ifs.close();
  std::cout << "Finish Reading Data" << '\n';
}
}  // namespace knn