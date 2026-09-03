//! Two properties of the sandbox that hold by construction, and would break quietly if they moved.
//!
//! Both were found by running DSPy's program-of-thought tutorial against both languages on one
//! machine: this crate's sandbox ran and dspy's did not, with a byte-identical `runner.js` and a
//! structurally identical `deno run` argv. The whole difference was *where the runner sits*.
//!
//! One test, not two: the staging directory is per-process, so a second test planting a file in it
//! would race the first reading it.

use dsrust::interpreter::{CodeInterpreter, DenoInterpreter};
use serde_json::Map;

#[test]
fn the_sandbox_resolves_pyodide_and_says_so_when_it_cannot() {
    let sandbox = DenoInterpreter::new();
    if sandbox.execute("1 + 1", &Map::new()).is_err() {
        eprintln!("no working deno on this machine; skipping");
        return;
    }
    let staged = std::env::temp_dir().join(format!("dsrust-sandbox-{}", std::process::id()));

    // Deno resolves `npm:pyodide` by walking up from the script for a `package.json`, and switches
    // to node_modules resolution when it finds one — where pyodide is not. Staging in temp is what
    // keeps a `package.json` in some ancestor of the *caller's* project from breaking the sandbox.
    assert!(
        staged.starts_with(std::env::temp_dir()),
        "{}",
        staged.display()
    );
    for ancestor in staged.ancestors() {
        assert!(
            !ancestor.join("package.json").exists(),
            "a package.json at {} would send deno to node_modules and lose pyodide",
            ancestor.display()
        );
    }

    // Plant exactly what the staging exists to avoid. dspy 3.3.1 stopped depending on the runner's
    // location for this: `DENO_NO_PACKAGE_JSON=1` and `--node-modules-dir=false` turn ambient
    // discovery off outright, so a `package.json` beside the runner no longer decides anything —
    // and the staging above stays because it is still what keeps two processes from handing each
    // other a half-written runner.
    let planted = staged.join("package.json");
    std::fs::write(&planted, r#"{"name":"planted","devDependencies":{}}"#).expect("writes");
    let ran = DenoInterpreter::new().execute("1 + 1", &Map::new());
    std::fs::remove_file(&planted).ok();
    assert_eq!(
        ran.expect("a package.json no longer reaches the sandbox's resolution"),
        dsrust::interpreter::Executed::Printed(serde_json::json!(2)),
    );

    // And the argv says so, rather than the run merely happening to work: the three flags dspy
    // 3.3.1 added are what make the plant above inert.
    let argv = DenoInterpreter::new().argv();
    for flag in ["--no-config", "--no-lock", "--node-modules-dir=false"] {
        assert!(
            argv.iter().any(|arg| arg == flag),
            "{flag} missing: {argv:?}"
        );
    }
}
