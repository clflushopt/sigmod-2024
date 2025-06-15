//! Constants used throughout the implementation.

/// Number of dimensions for the vector representation used in the datasets.
pub const VECTOR_DIMENSIONS: usize = 100;

/// K-nearest neighbors to return in the search results.
pub const K_NEAREST: usize = 100;

/// Attrbiute indices and dimensions used to encode the nodes in the dataset.
pub const NODE_C_ATTR_INDEX: usize = 0;
pub const NODE_T_ATTR_INDEX: usize = 1;
pub const NODE_VECTOR_START_INDEX: usize = 2;
pub const NODE_TOTAL_DIMENSIONS: usize = NODE_VECTOR_START_INDEX + VECTOR_DIMENSIONS;

/// Attribute indices and dimensions used to encode the queries in the dataset.
pub const QUERY_TYPE_INDEX: usize = 0;
pub const QUERY_V_CAT_INDEX: usize = 1;
pub const QUERY_T_LOWER_INDEX: usize = 2;
pub const QUERY_T_UPPER_INDEX: usize = 3;
pub const QUERY_VECTOR_START_INDEX: usize = 4;
pub const QUERY_TOTAL_DIMENSIONS: usize = QUERY_VECTOR_START_INDEX + VECTOR_DIMENSIONS;
