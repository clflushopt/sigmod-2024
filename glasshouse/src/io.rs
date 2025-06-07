//! Utilities for reading and writing the binary format used
//! by the SIGMOD 2024 datasets.
//!
//! The binary format used by the SIGMOD 2024 datasets is a custom format
//! that stores data points as vectors of floats. Each data point is stored
//! as a sequence of 4-byte floats in little-endian order.
//!
//! We differentiate between two types of files, nodes and queries.
//!
//! Nodes are represented as vectors of dimension `102` where the first
//! two entries in the vector are the categorical attribute and timestamp
//! attribute respectively. The remaining `100` entries are the actual vector
//! entries.
//!
//! Queries are represented as vectors of dimension `104` where the first
//! entry is the query type; which distinguishes between non-constrained
//! queries, equality queries, range queries and equality and range queries.
use crate::constants::*;
use crate::types::*; // Or specific types like NodesDataset, QueriesDataset, etc.
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::mem;
use std::path::Path;

impl NodesDataset {
    /// Returns a parsed node at the given index.
    pub fn get(&self, index: usize) -> Option<ParsedNode> {
        if index >= self.num_vectors as usize {
            return None;
        }
        Some(ParsedNode {
            c_attr: self.c_attrs[index],
            t_attr: self.t_attrs[index],
            vector: &self.vectors[index],
        })
    }

    /// Reads the nodes dataset from a binary file.
    pub fn read<P: AsRef<Path>>(file_path: P) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        let num_vectors = {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            u32::from_le_bytes(buf)
        };

        let mut c_attrs = Vec::with_capacity(num_vectors as usize);
        let mut t_attrs = Vec::with_capacity(num_vectors as usize);
        let mut vectors = Vec::with_capacity(num_vectors as usize);

        // Re-use a buffer for each item to avoid reallocations.
        let mut buffer = vec![0.0f32; NODE_TOTAL_DIMENSIONS];

        for _ in 0..num_vectors {
            // Unsafe block for reading directly into f32 slice
            unsafe {
                let byte_buffer = std::slice::from_raw_parts_mut(
                    buffer.as_mut_ptr() as *mut u8,
                    buffer.len() * mem::size_of::<f32>(),
                );
                reader.read_exact(byte_buffer)?;
            }

            c_attrs.push(buffer[NODE_C_ATTR_INDEX]);
            t_attrs.push(buffer[NODE_T_ATTR_INDEX]);

            let mut vector_data = [0.0f32; VECTOR_DIMENSIONS];
            vector_data.copy_from_slice(&buffer[NODE_VECTOR_START_INDEX..NODE_TOTAL_DIMENSIONS]);
            vectors.push(vector_data);
        }

        Ok(NodesDataset {
            num_vectors,
            c_attrs,
            t_attrs,
            vectors,
        })
    }
}

impl QueriesDataset {
    /// Returns a parsed query at the given index.
    pub fn get(&self, index: usize) -> Option<ParsedQuery> {
        if index >= self.num_queries as usize {
            return None;
        }
        Some(ParsedQuery {
            query_type: self.query_types[index],
            v_categorical: self.v_categoricals[index].categorical_value(),
            t_lower_bound: self.t_lower_bounds[index].value(),
            t_upper_bound: self.t_upper_bounds[index].value(),
            query_vector: &self.query_vectors[index],
        })
    }

    /// Reads the queries dataset from a binary file.
    pub fn read<P: AsRef<Path>>(file_path: P) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        let num_queries = {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            u32::from_le_bytes(buf)
        };

        let mut query_types_vec = Vec::with_capacity(num_queries as usize);
        let mut v_categoricals_vec = Vec::with_capacity(num_queries as usize);
        let mut t_lower_bounds_vec = Vec::with_capacity(num_queries as usize);
        let mut t_upper_bounds_vec = Vec::with_capacity(num_queries as usize);
        let mut query_vectors_vec = Vec::with_capacity(num_queries as usize);

        let mut buffer = vec![0.0f32; QUERY_TOTAL_DIMENSIONS];

        for _ in 0..num_queries {
            unsafe {
                let byte_buffer = std::slice::from_raw_parts_mut(
                    buffer.as_mut_ptr() as *mut u8,
                    buffer.len() * mem::size_of::<f32>(),
                );
                reader.read_exact(byte_buffer)?;
            }

            match QueryType::from_f32(buffer[QUERY_TYPE_INDEX]) {
                Ok(qt) => query_types_vec.push(qt),
                Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            }
            v_categoricals_vec.push(OptionalFilterValue::new(buffer[QUERY_V_CAT_INDEX]));
            t_lower_bounds_vec.push(OptionalFilterValue::new(buffer[QUERY_T_LOWER_INDEX]));
            t_upper_bounds_vec.push(OptionalFilterValue::new(buffer[QUERY_T_UPPER_INDEX]));

            let mut vector_data = [0.0f32; VECTOR_DIMENSIONS];
            vector_data.copy_from_slice(&buffer[QUERY_VECTOR_START_INDEX..QUERY_TOTAL_DIMENSIONS]);
            query_vectors_vec.push(vector_data);
        }

        Ok(QueriesDataset {
            num_queries,
            query_types: query_types_vec,
            v_categoricals: v_categoricals_vec,
            t_lower_bounds: t_lower_bounds_vec,
            t_upper_bounds: t_upper_bounds_vec,
            query_vectors: query_vectors_vec,
        })
    }
}

/// Saves the KNN results to a binary file.
/// The format is |Q| x K_NEAREST x id (uint32_t).
pub fn write<P: AsRef<Path>>(
    results: &QueryResults, // This is Vec<[u32; K_NEAREST]>
    file_path: P,
) -> io::Result<()> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    for single_query_results in results {
        // single_query_results is [u32; K_NEAREST]
        // Ensure K_NEAREST is correct (guaranteed by type)
        // Convert [u32; K_NEAREST] to &[u8] for writing
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                single_query_results.as_ptr() as *const u8,
                K_NEAREST * mem::size_of::<u32>(),
            )
        };
        writer.write_all(byte_slice)?;
    }
    writer.flush()?; // Ensure all buffered data is written
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_read_dummy_datasets() {
        let nodes_file = "tests/dummy-data.bin";
        let queries_file = "tests/dummy-queries.bin";
        // Read the file and check that it contains the expected data.
        let nodes = NodesDataset::read(nodes_file);
        let queries = QueriesDataset::read(queries_file);

        assert!(nodes.is_ok());
        assert!(queries.is_ok());

        assert!(nodes.unwrap().num_vectors == 10000);
        assert!(queries.unwrap().num_queries == 10000);
    }

    #[test]
    fn can_query_individual_nodes_and_queries() {
        let nodes_file = "tests/dummy-data.bin";
        let queries_file = "tests/dummy-queries.bin";

        let nodes = NodesDataset::read(nodes_file).unwrap();
        let queries = QueriesDataset::read(queries_file).unwrap();

        assert!(nodes.get(0).is_some());
        assert!(queries.get(0).is_some());
    }
}
