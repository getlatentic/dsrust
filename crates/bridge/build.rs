/// Supply the platform's extension-module link arguments (on macOS, `-undefined
/// dynamic_lookup`). Doing it here rather than in `.cargo/config.toml` keeps the flags on this
/// crate, where workspace-wide rustflags would also reach unrelated binaries and tests.
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
