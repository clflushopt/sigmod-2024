mod constants;
mod io;
mod types;

use std::{error::Error, time::Instant};

use constants::{K_NEAREST, VECTOR_DIMENSIONS};
use types::{NodesDataset, QueriesDataset, QueryResult, QueryResults, QueryType};

/// Baseline solution.
struct Baseline;

/// Calculates squared Euclidean distance between two 100-dim vectors.
fn l2(vec1: &[f32; VECTOR_DIMENSIONS], vec2: &[f32; VECTOR_DIMENSIONS]) -> f32 {
    vec1.iter().zip(vec2.iter()).fold(0.0, |acc, (a, b)| {
        let diff = a - b;
        acc + diff * diff
    })
}
impl Baseline {
    const DEFAULT_PAD_ID: u32 = 0; // Or u32::MAX
    const SAMPLE_PROPORTION: f32 = 0.001;

    pub fn run(
        nodes_dataset: &NodesDataset,
        queries_dataset: &QueriesDataset,
    ) -> Result<QueryResults, Box<dyn Error>> {
        let mut all_knn_results: QueryResults =
            Vec::with_capacity(queries_dataset.num_queries as usize);

        let num_to_sample = ((nodes_dataset.num_vectors as f32 * Self::SAMPLE_PROPORTION) as u32)
            .max(1) // Ensure at least 1 sample
            .min(nodes_dataset.num_vectors); // Don't exceed total number of nodes

        println!("Baseline Algorithm Parameters:");
        println!("  K-Nearest: {}", K_NEAREST);
        println!("  Sample proportion: {}", Self::SAMPLE_PROPORTION);
        println!("  Actual points to sample per query: {}", num_to_sample);

        for i in 0..(queries_dataset.num_queries as usize) {
            let query = queries_dataset
                .get(i)
                .ok_or_else(|| format!("Failed to get parsed query for index: {}", i))?;

            let mut qualified_candidates: Vec<(f32, u32)> = Vec::new();

            for node_idx in 0..num_to_sample {
                let node_id = node_idx; // In this sampling strategy, index is ID for the sampled prefix
                let node = nodes_dataset
                    .get(node_id as usize)
                    .ok_or_else(|| format!("Failed to get parsed node for ID: {}", node_id))?;

                let passes_filter = match query.query_type {
                    QueryType::VectorOnly => true,
                    QueryType::CategoricalConstraint => query
                        .v_categorical
                        .is_some_and(|v_cat| (node.c_attr - v_cat as f32).abs() < f32::EPSILON),
                    QueryType::TimestampConstraint => {
                        match (query.t_lower_bound, query.t_upper_bound) {
                            (Some(l_bound), Some(r_bound)) => {
                                node.t_attr >= l_bound && node.t_attr <= r_bound
                            }
                            _ => false,
                        }
                    }
                    QueryType::BothConstraints => {
                        let cat_match = query
                            .v_categorical
                            .is_some_and(|v_cat| (node.c_attr - v_cat as f32).abs() < f32::EPSILON);
                        let time_match = match (query.t_lower_bound, query.t_upper_bound) {
                            (Some(l_bound), Some(r_bound)) => {
                                node.t_attr >= l_bound && node.t_attr <= r_bound
                            }
                            _ => false,
                        };
                        cat_match && time_match
                    }
                };

                if passes_filter {
                    let dist = l2(query.query_vector, node.vector);
                    qualified_candidates.push((dist, node_id));
                }
            }

            qualified_candidates.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut current_knn_result: QueryResult = [Self::DEFAULT_PAD_ID; K_NEAREST];
            for k_idx in 0..K_NEAREST {
                if let Some(candidate) = qualified_candidates.get(k_idx) {
                    current_knn_result[k_idx] = candidate.1; // Store the ID
                }
            }
            all_knn_results.push(current_knn_result);
        }

        Ok(all_knn_results)
    }
}

fn main() {
    let program_start_time = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    let source_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("./tests/dummy-data.bin");
    let query_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("./tests/dummy-queries.bin");
    let knn_save_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("./tests/output.bin");

    // --- Data Loading ---
    let load_start_time = Instant::now();
    println!("[+] Loading nodes dataset from: {}", source_path);
    let nodes_dataset = NodesDataset::read(source_path);
    assert!(nodes_dataset.is_ok(), "Failed to load nodes dataset");
    let nodes_dataset = nodes_dataset.unwrap();
    println!(
        "[+] Loaded {} nodes in {:?}",
        nodes_dataset.num_vectors,
        load_start_time.elapsed()
    );

    let load_start_time = Instant::now();
    println!("[+] Loading queries dataset from: {}", query_path);
    let queries_dataset = QueriesDataset::read(query_path);
    assert!(queries_dataset.is_ok(), "Failed to load queries dataset");
    let queries_dataset = queries_dataset.unwrap();
    println!(
        "[+] Loaded {} queries in {:?}",
        queries_dataset.num_queries,
        load_start_time.elapsed()
    );

    // Run baseline solution.
    let algo_start_time = Instant::now();
    println!("[!] Running baseline solution...");
    let baseline_results = Baseline::run(&nodes_dataset, &queries_dataset);
    assert!(baseline_results.is_ok(), "Failed to run baseline algorithm");
    let baseline_results = baseline_results.unwrap();
    println!(
        "[*] Baseline solution completed in {:?}",
        algo_start_time.elapsed()
    );

    // Write results to disk.
    let save_start_time = Instant::now();
    println!("[*] Writing results to {}", knn_save_path);
    let _ = io::write(&baseline_results, knn_save_path);
    println!("[*] Writing results took {:?}", save_start_time.elapsed());

    let total_duration = program_start_time.elapsed();
    println!("[*] Total runtime was {:?}", total_duration);
}
