//! Utilities for reading and writing the binary format used
//! by the SIGMOD 2024 datasets.

type Vector = Vec<f32>;

/// Writes a binary file containing data points.
pub fn write(path: &str, data: &[Vector]) -> Result<(), std::io::Error> {
    use std::fs::File;
    use std::io::{self, Write};
    // The data points are stored as float vectors in a binary format.
    let mut file = File::create(path)?;

    // First 4 bytes are the number of data points.
    let n_points = data.len() as u32;
    file.write_all(&n_points.to_le_bytes())?;

    if n_points == 0 {
        return Ok(());
    }

    // Each data point is a vector of floats.
    for point in data {
        if point.len() != data[0].len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "All data points must have the same number of dimensions",
            ));
        }
        for &value in point {
            file.write_all(&value.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Reads a binary file containing data points.
pub fn read(path: &str, n_dims: usize) -> Result<Vec<Vector>, std::io::Error> {
    use std::fs::File;
    use std::io::{self, Read};

    println!("Processing file: {}", path);
    if n_dims == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Number of dimensions must be greater than zero",
        ));
    }

    let mut file = File::open(path)?;

    // First 4 bytes are the number of data points.
    let mut buffer = vec![0u8; 4];
    file.read_exact(&mut buffer)?;
    let n_points = u32::from_le_bytes(buffer.try_into().unwrap()) as usize;
    println!("Number of points: {}", n_points);
    if n_points == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "File contains no data points",
        ));
    }
    // The n-points are how many data points there are in total
    // and each data point is a vector of n_dims floats.
    //
    // We will read them row by row and write them to the `data`
    // vector as a vector of floats.
    let mut buffer = vec![0u8; n_points * n_dims * 4];
    file.read_exact(&mut buffer)?;
    if buffer.len() != n_points * n_dims * 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "File does not contain enough data points",
        ));
    }
    println!("Read {} bytes of data", buffer.len());

    // Next step is to convert the bytes into vectors of floats.
    // Each float is 4 bytes, so we can convert the bytes directly.
    let num_rows = n_points;
    let mut data = Vec::with_capacity(num_rows);
    let mut row = Vec::with_capacity(n_dims);
    for chunk in buffer.chunks_exact(n_dims * 4) {
        row.clear();
        for float_chunk in chunk.chunks_exact(4) {
            let float = f32::from_le_bytes(float_chunk.try_into().unwrap());
            row.push(float);
        }
        data.push(row.clone());
    }

    println!("Successfully read {} data points", data.len());
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read() {
        let nodes = "tests/dummy-data.bin";
        let queries = "tests/dummy-queries.bin";
        // Read the file and check that it contains the expected data.
        assert!(read(nodes, 102).is_ok());
        assert!(read(queries, 104).is_ok());
    }
}
