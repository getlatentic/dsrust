//! The `deno run` invocation, and where the runner it runs comes from.
//!
//! dspy builds this argv in `PythonInterpreter.__init__` and the permissions are the sandbox: what
//! is not granted cannot be reached. Deno matches `--allow-read` by string prefix against the
//! *realpath* of the file opened (denoland/deno#9607), so every path is resolved first or a read
//! through a symlink — including `DENO_DIR` — is denied.

use crate::error::Explained;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;

/// The sandbox itself: dspy's own `runner.js`, vendored so this crate runs the file upstream runs
/// rather than a reimplementation of it. MIT, Copyright (c) 2023 Stanford Future Data Systems.
const RUNNER: &str = include_str!("runner.js");

/// What the sandbox may reach beyond the runner and Pyodide's cache.
#[derive(Debug, Clone, Default)]
pub struct Permissions {
    /// Paths the sandboxed code may read. dspy's `enable_read_paths`.
    pub read: Vec<PathBuf>,
    /// Paths it may write. dspy's `enable_write_paths`; also granted read.
    pub write: Vec<PathBuf>,
    /// Hosts it may reach, as deno spells them. dspy's `enable_network_access`.
    pub network: Vec<String>,
    /// Environment variables it may see. dspy's `enable_env_vars`.
    pub env: Vec<String>,
}

/// Resolve symlinks, because that is what deno's permission check compares against.
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Where deno keeps its cache, so Pyodide's own files are readable.
///
/// dspy asks `deno info --json` for it rather than guessing at `~/.cache/deno`, since the location
/// moves with the platform and with `DENO_DIR`.
fn deno_dir() -> Option<PathBuf> {
    let asked = Command::new("deno")
        .args(["info", "--json"])
        .output()
        .ok()?;
    let info: serde_json::Value = serde_json::from_slice(&asked.stdout).ok()?;
    info.get("denoDir")
        .and_then(|dir| dir.as_str())
        .map(PathBuf::from)
}

/// Write the vendored runner where deno can read it, once per process.
///
/// It ships inside the binary, so there is no file to find at runtime and no install step for a
/// caller — which is the difference between a sandbox this crate has and one it describes.
///
/// The directory carries the process id because the write is a truncate: a second process sharing
/// the path could hand deno an empty runner to execute.
pub fn runner_path() -> Result<PathBuf> {
    let directory = std::env::temp_dir().join(format!("dsrust-sandbox-{}", std::process::id()));
    std::fs::create_dir_all(&directory).explain("making a place for the sandbox runner")?;
    let path = directory.join("runner.js");
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    if current != RUNNER {
        std::fs::write(&path, RUNNER).explain("writing the sandbox runner")?;
    }
    Ok(path)
}

/// The whole `deno run …` argv, in upstream's order.
pub fn argv(runner: &Path, permissions: &Permissions) -> Vec<String> {
    let runner = canonical(runner);
    let mut readable: Vec<PathBuf> = vec![runner.clone()];
    readable.extend(deno_dir());
    readable.extend(permissions.read.iter().map(|path| canonical(path)));
    readable.extend(permissions.write.iter().map(|path| canonical(path)));

    let mut args = vec![
        "run".to_owned(),
        format!("--allow-read={}", joined(&readable)),
    ];
    // The env argument is passed twice on purpose: once as a deno permission, and once as an
    // argument to the runner, which reads the list to decide what to expose to Python.
    let env = permissions.env.join(",");
    if !permissions.env.is_empty() {
        args.push(format!("--allow-env={env}"));
    }
    if !permissions.network.is_empty() {
        args.push(format!("--allow-net={}", permissions.network.join(",")));
    }
    if !permissions.write.is_empty() {
        let writable: Vec<PathBuf> = permissions
            .write
            .iter()
            .map(|path| canonical(path))
            .collect();
        args.push(format!("--allow-write={}", joined(&writable)));
    }
    args.push(runner.to_string_lossy().into_owned());
    if !permissions.env.is_empty() {
        args.push(env);
    }
    args
}

fn joined(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.to_string_lossy())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing is granted that was not asked for. A default sandbox reads the runner and deno's
    /// cache and no more — no network, no writes, no environment.
    #[test]
    fn a_default_sandbox_grants_only_what_it_must() {
        let args = argv(Path::new("/tmp/runner.js"), &Permissions::default());
        assert_eq!(args[0], "run");
        assert!(args[1].starts_with("--allow-read="), "{args:?}");
        assert!(
            !args.iter().any(|a| a.starts_with("--allow-net")),
            "{args:?}"
        );
        assert!(
            !args.iter().any(|a| a.starts_with("--allow-write")),
            "{args:?}"
        );
        assert!(
            !args.iter().any(|a| a.starts_with("--allow-env")),
            "{args:?}"
        );
        assert!(
            args.last().expect("a runner").ends_with("runner.js"),
            "{args:?}"
        );
    }

    /// A writable path is readable too — upstream adds it to both lists, since code that cannot
    /// read what it wrote cannot check its own work.
    #[test]
    fn a_writable_path_is_also_readable() {
        let permissions = Permissions {
            write: vec![PathBuf::from("/tmp/out")],
            ..Permissions::default()
        };
        let args = argv(Path::new("/tmp/runner.js"), &permissions);
        let read = args
            .iter()
            .find(|a| a.starts_with("--allow-read="))
            .expect("a read grant");
        assert!(read.contains("/tmp/out"), "{read}");
        assert!(
            args.iter().any(|a| a == "--allow-write=/tmp/out"),
            "{args:?}"
        );
    }

    /// The environment list is both a deno permission and an argument to the runner, which reads
    /// it to decide what to hand Python. Passing it once leaves the sandbox blind to its own grant.
    #[test]
    fn the_environment_list_is_passed_to_deno_and_to_the_runner() {
        let permissions = Permissions {
            env: vec!["HOME".to_owned(), "PATH".to_owned()],
            ..Permissions::default()
        };
        let args = argv(Path::new("/tmp/runner.js"), &permissions);
        assert!(
            args.iter().any(|a| a == "--allow-env=HOME,PATH"),
            "{args:?}"
        );
        assert_eq!(
            args.last().map(String::as_str),
            Some("HOME,PATH"),
            "{args:?}"
        );
    }

    /// The vendored runner is dspy's file, and it lands somewhere deno can open.
    #[test]
    fn the_runner_is_written_where_deno_can_read_it() {
        let path = runner_path().expect("the runner is written");
        let written = std::fs::read_to_string(&path).expect("readable");
        assert!(
            written.contains("pyodideModule"),
            "it is the pyodide runner"
        );
        assert_eq!(
            written, RUNNER,
            "and it is the vendored copy, byte for byte"
        );
    }
}
