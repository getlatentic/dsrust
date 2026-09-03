//! numpy's `.npy` file, for a float32 matrix: what `np.save` writes and `np.load` reads.
//!
//! The format is a magic string, a version, a little-endian header length, a Python dict literal
//! padded with spaces so the data starts on a 64-byte boundary, and the raw little-endian values.
//! `Embeddings::save` writes `corpus_embeddings.npy` this way so dspy loads the index back, and
//! `Embeddings::load` reads what dspy saved.

use anyhow::{Result, anyhow, bail};

const MAGIC: &[u8] = b"\x93NUMPY";
const ALIGNMENT: usize = 64;

/// The bytes `np.save` writes for a float32 matrix of `rows` rows.
pub fn encode_f32(rows: &[Vec<f32>]) -> Vec<u8> {
    let width = rows.first().map_or(0, Vec::len);
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        rows.len(),
        width
    );
    // Magic (6), version (2), header length (2), the dict, padding, and a newline — to a multiple
    // of 64 bytes altogether.
    let prefix = MAGIC.len() + 2 + 2;
    let padding = (ALIGNMENT - (prefix + dict.len() + 1) % ALIGNMENT) % ALIGNMENT;
    let header_len = dict.len() + padding + 1;
    let mut out = Vec::with_capacity(prefix + header_len + rows.len() * width * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', padding));
    out.push(b'\n');
    for row in rows {
        for value in row {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// The float32 matrix in a `.npy` file, as `np.load` reads it: C order, little-endian `<f4`.
pub fn decode_f32(bytes: &[u8]) -> Result<Vec<Vec<f32>>> {
    if !bytes.starts_with(MAGIC) {
        bail!("not a .npy file: the magic string is missing");
    }
    let major = bytes
        .get(6)
        .copied()
        .ok_or_else(|| anyhow!("truncated .npy header"))?;
    let (header_len, data_at) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        other => bail!("unsupported .npy version {other}"),
    };
    let header = std::str::from_utf8(&bytes[data_at..data_at + header_len])
        .map_err(|_| anyhow!(".npy header is not text"))?;
    if !header.contains("'<f4'") {
        bail!(".npy holds something other than little-endian float32: {header}");
    }
    if header.contains("'fortran_order': True") {
        bail!(".npy is in Fortran order, which this reader does not take");
    }
    let (rows, width) = shape_of(header)?;
    let data = &bytes[data_at + header_len..];
    if data.len() != rows * width * 4 {
        bail!(
            ".npy data is {} bytes where the shape needs {}",
            data.len(),
            rows * width * 4
        );
    }
    Ok(data
        .chunks_exact(width * 4)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect()
        })
        .collect())
}

/// `'shape': (3, 3)` — or `(3,)` for a vector, read as one row.
fn shape_of(header: &str) -> Result<(usize, usize)> {
    let start = header
        .find("'shape': (")
        .ok_or_else(|| anyhow!(".npy header states no shape: {header}"))?
        + "'shape': (".len();
    let end = header[start..]
        .find(')')
        .ok_or_else(|| anyhow!(".npy shape is unclosed: {header}"))?;
    let dims: Vec<usize> = header[start..start + end]
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| anyhow!(".npy shape dimension `{part}` is not a number"))
        })
        .collect::<Result<_>>()?;
    match dims.as_slice() {
        [rows, width] => Ok((*rows, *width)),
        [width] => Ok((1, *width)),
        _ => bail!(
            ".npy holds a {}-dimensional array, not a matrix",
            dims.len()
        ),
    }
}
