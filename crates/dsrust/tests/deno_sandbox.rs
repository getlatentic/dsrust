//! The Deno/Pyodide sandbox, running real Python.
//!
//! Ignored by default: it needs `deno` on the path, and the first run downloads Pyodide. Both are
//! the prerequisites dspy already asks of its own users.
//!
//! ```sh
//! cargo test --test deno_sandbox -- --ignored --nocapture --test-threads=1
//! ```

use dsrust::interpreter::{CodeInterpreter, DenoInterpreter, Executed, Permissions};
use serde_json::{Map, json};

fn sandbox() -> DenoInterpreter {
    assert!(
        DenoInterpreter::available(),
        "these tests need deno on the path"
    );
    DenoInterpreter::new()
}

/// One directory for both write-back tests, named for this process so a concurrent run of this
/// binary reads and writes its own copy rather than ours mid-assertion.
fn writeback_directory() -> std::path::PathBuf {
    let directory =
        std::env::temp_dir().join(format!("dsrust-writeback-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("a place to write");
    directory
}

/// Printed output is what the code wrote, which is what a module feeds back to the model.
#[test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
fn printed_output_comes_back() {
    let deno = sandbox();
    let ran = deno.execute("print(6 * 7)", &Map::new()).expect("it runs");
    assert_eq!(ran, Executed::Printed(json!("42\n")), "{ran:?}");
    deno.shutdown();
}

/// `SUBMIT` ends the episode, and the loop reads that differently from printed output.
///
/// The value arrives under `output` because that is what the sandbox's default `SUBMIT` writes —
/// `raise FinalOutput({"output": output})` — and a module unwraps it there. Asserting on the raw
/// shape is the point: it is upstream's, not a convenience this crate chose.
#[test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
fn submit_ends_the_episode_with_its_value() {
    let deno = sandbox();
    let ran = deno
        .execute("SUBMIT({'answer': 42})", &Map::new())
        .expect("it runs");
    assert!(ran.is_submitted(), "{ran:?}");
    assert_eq!(
        ran.value(),
        &json!({ "output": { "answer": 42 } }),
        "{ran:?}"
    );
    deno.shutdown();
}

/// Variables the caller passes are in scope, which is how RLM's inputs reach generated code.
#[test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
fn the_callers_variables_are_in_scope() {
    let deno = sandbox();
    let mut given = Map::new();
    given.insert("numbers".to_owned(), json!([1, 2, 3, 4]));
    given.insert("scale".to_owned(), json!(10));
    given.insert("on".to_owned(), json!(true));
    let ran = deno
        .execute("SUBMIT(sum(numbers) * scale if on else 0)", &given)
        .expect("it runs");
    assert_eq!(ran.value(), &json!({ "output": 100 }), "{ran:?}");
    deno.shutdown();
}

/// State persists between calls in one session, as upstream's does.
#[test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
fn a_name_bound_by_one_call_is_there_for_the_next() {
    let deno = sandbox();
    deno.execute("remembered = 7", &Map::new())
        .expect("it runs");
    let ran = deno
        .execute("SUBMIT(remembered * 3)", &Map::new())
        .expect("it runs");
    assert_eq!(ran.value(), &json!({ "output": 21 }), "{ran:?}");
    deno.shutdown();
}

/// The code's own failure is an error carrying Python's wording, because that text reaches the
/// model as the thing to correct.
#[test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
fn a_failure_carries_pythons_own_message() {
    let deno = sandbox();
    let refused = deno
        .execute("undefined_name + 1", &Map::new())
        .expect_err("it fails");
    let said = format!("{refused:#}");
    assert!(said.contains("undefined_name"), "{said}");
    deno.shutdown();
}

/// A file the sandbox writes appears on the host — but only after the sandbox is let go of.
///
/// `sync_file` is a notification already sitting in the pipe when the run returns, so a `shutdown`
/// that kills the child loses it. This test failed on exactly that, with an empty file.
#[test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
fn a_file_written_in_the_sandbox_reaches_the_host() {
    let directory = writeback_directory();
    let path = directory.join("written-inside.txt");
    std::fs::write(&path, "before").expect("the host's version");

    let deno = DenoInterpreter::permissions(Permissions {
        write: vec![path.clone()],
        ..Permissions::default()
    });
    let ran = deno
        .execute(
            "open('/sandbox/written-inside.txt', 'w').write('from inside')\nSUBMIT('done')",
            &Map::new(),
        )
        .expect("it runs");
    assert!(ran.is_submitted(), "{ran:?}");
    deno.shutdown();

    assert_eq!(
        std::fs::read_to_string(&path).expect("readable"),
        "from inside"
    );
    let _ = std::fs::remove_file(&path);
}

/// And stays on the host's version when write-back is off, which is the other half of the flag.
#[test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
fn without_write_back_the_hosts_file_is_left_alone() {
    let directory = writeback_directory();
    let path = directory.join("left-alone.txt");
    std::fs::write(&path, "untouched").expect("the host's version");

    let deno = DenoInterpreter::permissions(Permissions {
        write: vec![path.clone()],
        ..Permissions::default()
    })
    .without_write_back();
    deno.execute(
        "open('/sandbox/left-alone.txt', 'w').write('ignored')\nSUBMIT('done')",
        &Map::new(),
    )
    .expect("it runs");
    deno.shutdown();

    assert_eq!(
        std::fs::read_to_string(&path).expect("readable"),
        "untouched"
    );
    let _ = std::fs::remove_file(&path);
}
