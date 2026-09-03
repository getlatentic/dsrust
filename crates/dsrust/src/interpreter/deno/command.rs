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

/// dspy 3.3.1's `DENO_PROBE_TIMEOUT_SECONDS`: how long a `deno info` or `deno --version` may take.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// dspy 3.3.1's `_deno_subprocess_env`: the child and every probe run with ambient `package.json`
/// discovery off, so a project file above the runner cannot switch deno to node resolution.
pub(super) fn deno_env(command: &mut Command) -> &mut Command {
    command.env("DENO_NO_PACKAGE_JSON", "1")
}

/// A probe's stdout, or nothing when deno is missing, fails, or outlasts the timeout.
fn probe(args: &[&str]) -> Option<String> {
    let mut child = deno_env(&mut Command::new("deno"))
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let started = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = std::io::Read::read_to_string(&mut stdout, &mut out);
                }
                return status.success().then_some(out);
            }
            Ok(None) if started.elapsed() < PROBE_TIMEOUT => {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// Where deno keeps its cache, so Pyodide's own files are readable.
///
/// dspy reads `DENO_DIR` first and otherwise asks `deno info --json`, rather than guessing at
/// `~/.cache/deno`, since the location moves with the platform.
pub(super) fn deno_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("DENO_DIR") {
        return Some(PathBuf::from(dir));
    }
    let info: serde_json::Value = serde_json::from_str(&probe(&["info", "--json"])?).ok()?;
    info.get("denoDir")
        .and_then(|dir| dir.as_str())
        .map(PathBuf::from)
}

/// dspy 3.3.1's `MIN_DENO_VERSION`/`MAX_DENO_VERSION`: the sandbox runs on Deno 2.
const MIN_DENO: (u64, u64, u64) = (2, 0, 0);
const MAX_DENO: (u64, u64, u64) = (3, 0, 0);

/// `deno --version`'s first line as a version, or nothing when it does not start `deno X.Y.Z`.
pub(super) fn parse_version(output: &str) -> Option<(u64, u64, u64)> {
    let rest = output.strip_prefix("deno ")?;
    let spelled = rest
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .next()?;
    let mut parts = spelled.split('.').map(str::parse::<u64>);
    let version = (
        parts.next()?.ok()?,
        parts.next()?.ok()?,
        parts.next()?.ok()?,
    );
    Some(version)
}

/// dspy 3.3.1's `_validate_deno_version`, in its words: refuse a deno that is not 2.x before it is
/// asked to run anything. `pip install "dspy[deno]"` is upstream's remedy; the one that applies here
/// is a Deno 2 install.
pub fn validate_version(version: Option<(u64, u64, u64)>) -> Result<()> {
    let remedy = "Install a compatible Deno 2 (`curl -fsSL https://deno.land/install.sh | sh`, or \
                  `brew install deno`).";
    match version {
        None => anyhow::bail!(
            "Unable to determine the Deno version from \"deno\". PythonInterpreter requires Deno \
             >=2.0.0,<3.0.0. {remedy}"
        ),
        Some(version) if version < MIN_DENO || version >= MAX_DENO => anyhow::bail!(
            "Unsupported Deno version {}.{}.{}. PythonInterpreter supports Deno >=2.0.0,<3.0.0. \
             {remedy}",
            version.0,
            version.1,
            version.2
        ),
        Some(_) => Ok(()),
    }
}

/// The installed deno's version, probed on each call.
///
/// dspy 3.3.1 asks the runtime every time it is about to spawn one, and says so in the name of
/// `test_deno_version_probe_is_bounded_and_not_cached`. A remembered answer would keep refusing a
/// deno the user has since upgraded, for as long as the process lives.
pub(super) fn installed_version() -> Option<(u64, u64, u64)> {
    parse_version(&probe(&["--version"])?)
}

/// dspy 3.3.1's `_paths_overlap`: the same path, or one inside the other. A trailing separator
/// does not change the answer, and neither does a shared prefix that stops mid-component —
/// `/tmp/deno-cache` does not overlap `/tmp/deno`.
pub fn paths_overlap(first: &Path, second: &Path) -> bool {
    first == second || first.starts_with(second) || second.starts_with(first)
}

/// dspy 3.3.1: a writable path may not reach the runner or deno's cache, which the sandbox reads
/// its own runtime from.
pub(super) fn refuse_overlapping_writes(runner: &Path, permissions: &Permissions) -> Result<()> {
    let mut protected = vec![runner.to_path_buf()];
    protected.extend(deno_dir());
    refuse_writes_over(&protected, &permissions.write)
}

/// The same rule over paths the caller names.
pub(super) fn refuse_writes_over(protected: &[PathBuf], write: &[PathBuf]) -> Result<()> {
    let protected: Vec<PathBuf> = protected.iter().map(|path| canonical(path)).collect();
    let overlapping = write
        .iter()
        .map(|path| canonical(path))
        .any(|path| protected.iter().any(|kept| paths_overlap(&path, kept)));
    match overlapping {
        true => anyhow::bail!("Write paths cannot overlap PythonInterpreter runtime files."),
        false => Ok(()),
    }
}

/// Write the vendored runner where deno can read it, once per process.
///
/// It ships inside the binary, so there is no file to find at runtime and no install step for a
/// caller — which is the difference between a sandbox this crate has and one it describes.
///
/// The directory carries the process id because the write is a truncate: a second process sharing
/// the path could hand deno an empty runner to execute.
///
/// **Temp is load-bearing, not merely convenient.** Deno resolves a `npm:` import by walking *up*
/// from the script for a `package.json`, and switches to node_modules resolution the moment it
/// finds one — whatever that file is for. Pyodide lives in deno's global cache and not in that
/// `node_modules`, so the sandbox dies at startup with "Could not find a matching package for
/// 'npm:pyodide'". Staging here puts the runner under a path with no `package.json` above it,
/// which is why this crate's sandbox runs where dspy's own does not on the same machine: dspy runs
/// its runner from inside the installed package, and a `package.json` in a home directory two
/// levels up is enough to break it. Moving the runner back beside the source would reintroduce
/// that, silently, and only for callers whose parents happen to hold one.
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

/// The whole `deno run …` argv, in upstream's order. dspy 3.3.1 runs the sandbox with no config
/// file, no lockfile and no `node_modules` resolution, hands the runner the env list on every run —
/// empty or not — and names deno's cache so the runner can revoke its own read of it once Pyodide
/// has loaded.
pub fn argv(runner: &Path, permissions: &Permissions) -> Vec<String> {
    let runner = canonical(runner);
    let deno_dir = deno_dir();
    let mut readable: Vec<PathBuf> = vec![runner.clone()];
    readable.extend(deno_dir.clone());
    readable.extend(permissions.read.iter().map(|path| canonical(path)));
    readable.extend(permissions.write.iter().map(|path| canonical(path)));

    let mut args = vec![
        "run".to_owned(),
        "--no-config".to_owned(),
        "--no-lock".to_owned(),
        "--node-modules-dir=false".to_owned(),
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
    args.push(env);
    if let Some(dir) = deno_dir {
        args.push(format!(
            "--dspy-deno-dir={}",
            canonical(&dir).to_string_lossy()
        ));
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
        assert!(args[4].starts_with("--allow-read="), "{args:?}");
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
        let runner_at = args
            .iter()
            .position(|a| a.ends_with("runner.js"))
            .expect("a runner");
        assert_eq!(args[runner_at + 1], "", "an empty env list follows it");
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
        let runner_at = args
            .iter()
            .position(|a| a.ends_with("runner.js"))
            .expect("a runner");
        assert_eq!(args[runner_at + 1], "HOME,PATH", "{args:?}");
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

    /// dspy 3.3.1's `_deno_subprocess_env`: every deno this crate starts — the probes included —
    /// is told to ignore a `package.json`, so a manifest beside the working directory cannot add
    /// dependencies to the sandbox's runtime.
    #[test]
    fn every_deno_is_told_to_ignore_a_package_manifest() {
        let mut command = Command::new("deno");
        let set: Vec<_> = deno_env(&mut command)
            .get_envs()
            .map(|(name, value)| {
                (
                    name.to_string_lossy().into_owned(),
                    value.map(|value| value.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert_eq!(
            set,
            vec![("DENO_NO_PACKAGE_JSON".to_owned(), Some("1".to_owned()))]
        );
    }

    /// dspy 3.3.1's argv: the three isolation flags lead, the env list is always handed over, and
    /// deno's cache is named for the runner to revoke.
    #[test]
    fn the_isolation_flags_lead_and_the_env_list_is_always_passed() {
        let args = argv(Path::new("/tmp/runner.js"), &Permissions::default());
        assert_eq!(
            &args[..4],
            [
                "run",
                "--no-config",
                "--no-lock",
                "--node-modules-dir=false"
            ]
        );
        let runner_at = args
            .iter()
            .position(|a| a.ends_with("runner.js"))
            .expect("the runner");
        assert_eq!(
            args[runner_at + 1],
            "",
            "the env list, empty, follows the runner"
        );
        if deno_dir().is_some() {
            assert!(
                args[runner_at + 2].starts_with("--dspy-deno-dir="),
                "{args:?}"
            );
        }
    }

    #[test]
    fn a_version_is_read_off_denos_first_line() {
        assert_eq!(
            parse_version("deno 2.4.1 (stable, release, aarch64-apple-darwin)\nv8 13"),
            Some((2, 4, 1))
        );
        assert_eq!(parse_version("deno 1.46.3 (stable)"), Some((1, 46, 3)));
        assert_eq!(parse_version("something else"), None);
    }

    /// The refusals in upstream's words, for a deno that cannot be read and one outside 2.x.
    #[test]
    fn only_deno_2_is_accepted() {
        assert!(validate_version(Some((2, 0, 0))).is_ok());
        assert!(validate_version(Some((2, 9, 9))).is_ok());
        let unreadable = validate_version(None).expect_err("refused").to_string();
        assert!(unreadable.starts_with("Unable to determine the Deno version from \"deno\". PythonInterpreter requires Deno >=2.0.0,<3.0.0."), "{unreadable}");
        let old = validate_version(Some((1, 46, 3)))
            .expect_err("refused")
            .to_string();
        assert!(
            old.starts_with(
                "Unsupported Deno version 1.46.3. PythonInterpreter supports Deno >=2.0.0,<3.0.0."
            ),
            "{old}"
        );
        assert!(validate_version(Some((3, 0, 0))).is_err());
    }

    /// A writable path inside the runner's directory would let the sandbox rewrite its own runtime.
    #[test]
    fn a_write_path_over_the_runtime_is_refused() {
        let runner = std::env::temp_dir()
            .join("dsrust-overlap-test")
            .join("runner.js");
        let permissions = Permissions {
            write: vec![runner.parent().expect("a directory").to_path_buf()],
            ..Permissions::default()
        };
        let refused = refuse_overlapping_writes(&runner, &permissions).expect_err("refused");
        assert_eq!(
            refused.to_string(),
            "Write paths cannot overlap PythonInterpreter runtime files."
        );
        assert!(refuse_overlapping_writes(&runner, &Permissions::default()).is_ok());
    }
}
