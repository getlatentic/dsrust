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

    // Plant exactly what the staging exists to avoid, and the sandbox should fail — which is the
    // proof that the staging is load-bearing rather than merely tidy.
    let planted = staged.join("package.json");
    std::fs::write(&planted, r#"{"name":"planted","devDependencies":{}}"#).expect("writes");
    let failed = DenoInterpreter::new()
        .execute("1 + 1", &Map::new())
        .expect_err("a package.json beside the runner breaks pyodide resolution");
    std::fs::remove_file(&planted).ok();

    // And it should explain itself. The RPC layer sees only a closed pipe; deno's reason is on its
    // stderr, and a caller told "the sandbox closed its output" has nothing to act on where dspy
    // names the missing package.
    let said = failed.to_string();
    assert!(said.contains("Deno said:"), "dropped deno's reason: {said}");
    assert!(said.contains("pyodide"), "should name what failed: {said}");

    // The sandbox works again once the plant is gone, so the test leaves nothing behind.
    assert!(
        DenoInterpreter::new().execute("1 + 1", &Map::new()).is_ok(),
        "removing the plant should restore the sandbox"
    );
}
