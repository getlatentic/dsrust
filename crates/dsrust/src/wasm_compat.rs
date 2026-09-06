//! Concurrency bounds shared by native executors and single-threaded WebAssembly hosts.
//!
//! Cloudflare Workers execute Rust on `wasm32-unknown-unknown`, where browser-style HTTP
//! futures are intentionally `!Send`. Native callers still receive the original `Send + Sync`
//! guarantees. WASI is deliberately not included: relaxing its bounds would claim support for a
//! different host model that dsrust does not test.
//!
//! `dyn Future + WasmCompatSend` is rejected by the compiler — only auto traits may be added to a
//! trait object — so the boxed forms are aliases chosen per target rather than one alias carrying
//! a marker. The bound positions, where `impl Trait` accepts any trait, use the markers.
//!
//! Each target defines every name exactly once. Alternating `#[cfg]` at file scope defines each
//! twice, which reads as one trait owning the next as an associated item in any reader that does
//! not evaluate `cfg` — `scripts/rust_surface.py` among them.

use std::future::Future;
use std::pin::Pin;

use futures_util::Stream;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod bounds {
    use super::{Future, Pin, Stream};

    /// `Send`, on a target with threads to send between.
    pub trait WasmCompatSend: Send {}
    impl<T: Send + ?Sized> WasmCompatSend for T {}

    /// `Sync`, on a target with threads to share across.
    pub trait WasmCompatSync: Sync {}
    impl<T: Sync + ?Sized> WasmCompatSync for T {}

    pub type WasmBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
    pub type WasmBoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod bounds {
    use super::{Future, Pin, Stream};

    /// Unbounded: a Worker runs one thread, so nothing can leave it.
    pub trait WasmCompatSend {}
    impl<T: ?Sized> WasmCompatSend for T {}

    /// Unbounded, for the same reason.
    pub trait WasmCompatSync {}
    impl<T: ?Sized> WasmCompatSync for T {}

    pub type WasmBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;
    pub type WasmBoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + 'a>>;
}

pub use bounds::{WasmBoxFuture, WasmBoxStream, WasmCompatSend, WasmCompatSync};
