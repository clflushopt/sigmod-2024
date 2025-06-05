/**
 *  Example code using sampling to find KNN.
 *
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "io.h"

using std::cout;
using std::string;
using std::vector;

static auto compare_with_id(const std::vector<float> &lhs,
                            const std::vector<float> &rhs) -> float {
  float sum = 0.0;
  // Skip the first 2 dimensions
  for (size_t i = 2; i < lhs.size(); ++i) {
    float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

struct PointsAndQueries {
  vector<vector<float>> nodes;
  vector<vector<float>> queries;
};

static auto read_points_and_queries(const string &source_path,
                                    const string &query_path,
                                    const uint32_t node_dimensions,
                                    const uint32_t query_dimensions)
    -> PointsAndQueries {
  PointsAndQueries pq;
  knn::ReadBin(source_path, node_dimensions,
               pq.nodes);  // Assuming 102 dimensions
  knn::ReadBin(query_path, query_dimensions,
               pq.queries);  // Assuming queries have 104 dimensions
  return pq;
}

auto main(int argc, char *argv[]) -> int {
  string source_path = "./tests/dummy-data.bin";
  string query_path = "./tests/dummy-queries.bin";
  string knn_save_path = "./tests/output.bin";

  static constexpr uint32_t kNumDataDimensions = 102;
  // kNumQueryDimensions = kNumDataDimensions + 2
  static constexpr uint32_t kNumQueryDimensions = 104;
  static constexpr float kSampleProportion = 0.001F;

  // Also accept other path for source data
  if (argc > 1) {
    source_path = string(argv[1]);
  }

  PointsAndQueries pq = read_points_and_queries(
      source_path, query_path, kNumDataDimensions, kNumQueryDimensions);

  vector<vector<uint32_t>> knn_results;  // for saving knn results

  auto [nodes, queries] = std::tie(pq.nodes, pq.queries);

  uint32_t n_points = pq.nodes.size();
  uint32_t dimensions = pq.nodes[0].size();
  uint32_t n_queries = pq.queries.size();
  auto sample_prop = (n_points * static_cast<uint32_t>(kSampleProportion));

  cout << "# data points:  " << n_points << "\n";
  cout << "# data point dim:  " << dimensions << "\n";
  cout << "# queries:      " << n_queries << "\n";

  /** A basic method to compute the KNN results using sampling  **/
  const int K_NEAREST = 100;  // To find 100-NN

  for (uint i = 0; i < n_queries; i++) {
    auto query_type = static_cast<uint32_t>(queries[i][0]);
    auto value = static_cast<int32_t>(queries[i][1]);
    float l_bound = queries[i][2];
    float r_bound = queries[i][3];
    vector<float> query_vec;

    // first push_back 2 zeros for aligning with dataset
    query_vec.push_back(0);
    query_vec.push_back(0);
    for (uint j = 4; j < queries[i].size(); j++) {
      query_vec.push_back(queries[i][j]);
    }

    vector<uint32_t> knn;  // candidate knn

    // Handling 4 types of queries
    if (query_type == 0) {  // only ANN
      for (uint32_t j = 0; j < sample_prop; j++) {
        knn.push_back(j);
      }
    } else if (query_type == 1) {  // equal + ANN
      for (uint32_t j = 0; j < sample_prop; j++) {
        if (nodes[j][0] == static_cast<float>(value)) {
          knn.push_back(j);
        }
      }
    } else if (query_type == 2) {  // range + ANN
      for (uint32_t j = 0; j < sample_prop; j++) {
        if (nodes[j][1] >= l_bound && nodes[j][1] <= r_bound) {
          knn.push_back(j);
        }
      }
    } else if (query_type == 3) {  // equal + range + ANN
      for (uint32_t j = 0; j < sample_prop; j++) {
        if (nodes[j][0] == static_cast<float>(value) &&
            nodes[j][1] >= l_bound && nodes[j][1] <= r_bound) {
          knn.push_back(j);
        }
      }
    }

    // If the number of knn in the sampled data is less than K, then fill the
    // rest with the last few nodes
    if (knn.size() < K_NEAREST) {
      uint32_t sampled = 1;
      while (knn.size() < K_NEAREST) {
        knn.push_back(n_points - sampled);
        sampled = sampled + 1;
      }
    }

    // build another vec to store the distance between knn[i] and query_vec
    vector<float> dists;
    dists.resize(knn.size());
    for (uint32_t j = 0; j < knn.size(); j++) {
      dists[j] = compare_with_id(nodes[knn[j]], query_vec);
    }

    vector<uint32_t> ids;
    ids.resize(knn.size());
    std::iota(ids.begin(), ids.end(), 0);
    // sort ids based on dists
    std::ranges::sort(ids, [&](uint32_t lhs, uint32_t rhs) {
      return dists[lhs] < dists[rhs];
    });
    vector<uint32_t> knn_sorted;
    knn_sorted.resize(K_NEAREST);
    for (uint32_t j = 0; j < K_NEAREST; j++) {
      knn_sorted[j] = knn[ids[j]];
    }
    knn_results.push_back(knn_sorted);
  }

  // save the results
  knn::SaveKNN(knn_results, knn_save_path);
  return 0;
}