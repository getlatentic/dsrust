# Engineering standards

## Plan before implementation

- Establish the user story and acceptance criteria. Ask when missing information changes the required behavior.
- Describe the entry-to-outcome data flow, ownership, domain boundaries, and API contracts.
- Outline or write tests for the expected behavior before changing the implementation.

## Code for the next reader

- Make control flow and data flow understandable by scanning the code.
- Use precise domain names and small, cohesive abstractions that a junior developer can follow.
- Prefer less code and fewer branches when they preserve correctness, completeness, and readable intent.
- Use names and structure to explain what code does. Reserve comments for invariants, safety requirements, and reasons the code cannot express.
- Apply SOLID, KISS, DRY, and YAGNI with judgment. Use established patterns when they fit a concrete need.
- Organize by domain. Split modules that accumulate unrelated responsibilities.

## Preserve explicit behavior

- Build production implementations that fulfill the entire agreed scope.
- Do not add speculative fallbacks, compatibility branches, silent error handling, or workaround logic.
- Represent required absence explicitly. Report invalid input and broken invariants instead of substituting plausible output.
- Refactor broken abstractions instead of layering patches over them.
- Own defects discovered during the work: reproduce them, fix them, and add appropriate regression coverage.
- In Rust, make ownership, pinning, cancellation, destruction, and synchronization requirements explicit. Avoid unnecessary allocations and cloning without weakening those guarantees.

## Verify before claiming completion

- Review every change for correctness, readability, completeness, complexity, memory use, and performance.
- Run focused regression tests, then the repository checks required by the change.
- Report the checks actually run and their results. Distinguish verified behavior from untested assumptions.
- Never approve code merely because it compiles or existing tests pass.

`AGENTS.md` is a relative symlink to this file so Codex and Claude read the same standards.
