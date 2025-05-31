/**
 *  Example code for IO, read binary data vectors and save KNNs to path.
 *
 */

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace knn {
/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>> &knns,
             const std::string &path = "../output.bin");

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path, int num_dimensions,
             std::vector<std::vector<float>> &data);
}  // namespace knn
