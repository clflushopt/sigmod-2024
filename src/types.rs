//! Types used to represent data points and queries for the solvers.
use crate::constants::*;

/// Possible type of queries that can be made against the dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    VectorOnly,
    CategoricalConstraint,
    TimestampConstraint,
    BothConstraints,
}

impl QueryType {
    pub fn from_f32(val: f32) -> Result<Self, String> {
        // The query type is represented as a float but guaranteed to be one
        // of (0,1,2,3) so this cast is safe.
        let int_val = val as i32;
        match int_val {
            0 => Ok(QueryType::VectorOnly),
            1 => Ok(QueryType::CategoricalConstraint),
            2 => Ok(QueryType::TimestampConstraint),
            3 => Ok(QueryType::BothConstraints),
            _ => Err(format!("Invalid query type value: {}", val)),
        }
    }
}

/// Wrapper for attribute filter values that can be "not set" (represented by -1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionalFilterValue(f32);

impl OptionalFilterValue {
    pub fn new(val: f32) -> Self {
        OptionalFilterValue(val)
    }

    /// Returns the value if it's set, otherwise None.
    pub fn value(&self) -> Option<f32> {
        if self.0 == -1.0 { None } else { Some(self.0) }
    }

    pub fn categorical_value(&self) -> Option<i32> {
        if self.0 == -1.0 {
            None
        } else {
            Some(self.0 as i32)
        }
    }
}

#[derive(Debug, Default)]
pub struct NodesDataset {
    pub num_vectors: u32,
    /// Discretized categorical attribute C for each vector.
    pub c_attrs: Vec<f32>,
    /// Normalized timestamp attribute T for each vector.
    pub t_attrs: Vec<f32>,
    /// The 100-dimensional vectors.
    pub vectors: Vec<[f32; VECTOR_DIMENSIONS]>,
}

#[derive(Debug, Default)]
pub struct QueriesDataset {
    pub num_queries: u32,
    /// Type of each query.
    pub query_types: Vec<QueryType>,
    /// Specific query value v for the categorical attribute.
    pub v_categoricals: Vec<OptionalFilterValue>,
    /// Specific query value l for the timestamp attribute.
    pub t_lower_bounds: Vec<OptionalFilterValue>,
    /// Specific query value r for the timestamp attribute.
    pub t_upper_bounds: Vec<OptionalFilterValue>,
    /// The 100-dimensional query vectors.
    pub query_vectors: Vec<[f32; VECTOR_DIMENSIONS]>,
}

/// Represents a single parsed query with its associated attributes.
#[derive(Debug)]
pub struct ParsedQuery<'a> {
    pub query_type: QueryType,
    pub v_categorical: Option<i32>,
    pub t_lower_bound: Option<f32>,
    pub t_upper_bound: Option<f32>,
    pub query_vector: &'a [f32; VECTOR_DIMENSIONS],
}

/// Represents a single node with it's associated attributes.
#[derive(Debug)]
pub struct ParsedNode<'a> {
    pub c_attr: f32,
    pub t_attr: f32,
    pub vector: &'a [f32; VECTOR_DIMENSIONS],
}

/// Type alias for the KNN results for a single query.
pub type QueryResult = [u32; K_NEAREST];
/// Type alias for all KNN results.
pub type QueryResults = Vec<QueryResult>;
