//! dspy `_mount_files` / `_sync_files`: the host's files, in and back out of the sandbox.
//!
//! Pyodide has its own in-memory filesystem, so a path granted by `--allow-read` is still not a path
//! the sandboxed Python can open. Mounting copies each one in under `/sandbox/<basename>` — the
//! basename, so code refers to the file by the name the caller passed rather than by a host path it
//! has no reason to know.
//!
//! Syncing is the other direction, and it is a notification rather than a request: upstream writes
//! it and does not wait, so nothing here reads for a reply that never comes.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::json;

/// The name sandboxed code opens, for a host path.
fn virtual_path(path: &Path) -> String {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    format!("/sandbox/{name}")
}

/// Resolve symlinks, so the path matches the grant deno checked.
fn host_path(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// One `(host, virtual)` pair per file to mount, in upstream's order: reads then writes.
///
/// A writable path that does not exist yet is created empty, because code told it may write there
/// should not have to check first. A *readable* path that does not exist is an error — the caller
/// named a file they meant to supply.
pub(super) fn to_mount(read: &[PathBuf], write: &[PathBuf]) -> Result<Vec<(String, String)>> {
    let mut mounted = Vec::new();
    for path in read.iter().chain(write) {
        if path.as_os_str().is_empty() {
            continue;
        }
        if !path.exists() {
            if !write.contains(path) {
                bail!("Cannot mount non-existent file: {}", path.display());
            }
            std::fs::File::options()
                .append(true)
                .create(true)
                .open(path)
                .with_context(|| format!("creating {} to mount it", path.display()))?;
        }
        mounted.push((host_path(path), virtual_path(path)));
    }
    Ok(mounted)
}

/// The `mount_file` params for one pair.
pub(super) fn mount_request(host: &str, virtual_at: &str) -> serde_json::Value {
    json!({ "host_path": host, "virtual_path": virtual_at })
}

/// The `sync_file` params for each writable path, to copy the sandbox's version back out.
pub(super) fn to_sync(write: &[PathBuf]) -> Vec<serde_json::Value> {
    write
        .iter()
        .map(|path| json!({ "virtual_path": virtual_path(path), "host_path": host_path(path) }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sandbox sees the basename, not the host's directory layout — code that was handed
    /// `notes.txt` opens `/sandbox/notes.txt` wherever the file actually lives.
    #[test]
    fn the_sandbox_sees_the_basename() {
        assert_eq!(
            virtual_path(Path::new("/var/data/notes.txt")),
            "/sandbox/notes.txt"
        );
    }

    /// A readable file that is not there is the caller's mistake, and saying which one is missing is
    /// the whole value of the message.
    #[test]
    fn a_missing_readable_file_is_refused_by_name() {
        let refused =
            to_mount(&[PathBuf::from("/nonexistent/absent.txt")], &[]).expect_err("refused");
        assert_eq!(
            refused.to_string(),
            "Cannot mount non-existent file: /nonexistent/absent.txt"
        );
    }

    /// A writable one is created instead, so code told it may write does not have to check first.
    #[test]
    fn a_missing_writable_file_is_created() {
        let directory =
            std::env::temp_dir().join(format!("dsrust-mount-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("a place to write");
        let path = directory.join("created-on-mount.txt");
        let _ = std::fs::remove_file(&path);

        let mounted = to_mount(&[], std::slice::from_ref(&path)).expect("mounts");
        assert!(path.exists(), "the file was created");
        assert_eq!(mounted.len(), 1);
        assert!(
            mounted[0].1.ends_with("/sandbox/created-on-mount.txt"),
            "{mounted:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Only writable paths sync back. A readable one syncing would overwrite the host's file with
    /// the sandbox's copy, which is the opposite of read-only.
    #[test]
    fn only_writable_paths_sync_back() {
        assert!(to_sync(&[]).is_empty());
        let asked = to_sync(&[PathBuf::from("/tmp/out.txt")]);
        assert_eq!(asked[0]["virtual_path"], json!("/sandbox/out.txt"));
    }
}
