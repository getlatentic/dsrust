//! The bounds and the request-local usage scope, exercised through the paths a consumer imports.
//!
//! Both are reached here by their public names rather than their defining modules. The unit tests
//! beside each one prove the mechanism; these prove the re-export, which is the half a consumer
//! actually touches and the half a wrong `pub use` would break silently.

use std::sync::Arc;

use dsrust::lm::{LmUsage, UsageScope, UsageTracker, scoped_usage};
use dsrust::wasm_compat::{WasmBoxFuture, WasmBoxStream, WasmCompatSend, WasmCompatSync};

fn needs_send<T: WasmCompatSend>() {}
fn needs_sync<T: WasmCompatSync>() {}

/// On a target with threads the markers must still be `Send` and `Sync` — the whole point of the
/// split is that native callers keep the guarantee they had. A marker that resolved to nothing
/// everywhere would compile just as happily, so this is the assertion that tells the two apart.
#[test]
fn the_native_bounds_are_the_real_ones() {
    needs_send::<Arc<UsageTracker>>();
    needs_sync::<Arc<UsageTracker>>();

    let future: WasmBoxFuture<'static, u8> = Box::pin(async { 7u8 });
    let stream: WasmBoxStream<'static, u8> = Box::pin(futures_util::stream::iter([1u8, 2]));
    needs_send::<WasmBoxFuture<'static, u8>>();
    needs_send::<WasmBoxStream<'static, u8>>();
    drop((future, stream));
}

/// Two scopes over one thread, each charged while the other is suspended. A tracker installed for
/// the whole `await` rather than for each poll would let the second scope's charge land in the
/// first, which is the failure the type exists to prevent.
#[tokio::test]
async fn interleaved_scopes_keep_their_own_totals_through_the_public_path() {
    let first: UsageScope = scoped_usage();
    let second = scoped_usage();

    let charge = |scope: &UsageScope, tokens: u32| {
        let scope = scope.clone();
        async move {
            scope
                .run(async {
                    tokio::task::yield_now().await;
                    scope.tracker().add("m", LmUsage::counted(tokens, tokens));
                    tokio::task::yield_now().await;
                })
                .await;
        }
    };
    tokio::join!(charge(&first, 3), charge(&second, 5));

    assert_eq!(first.total().input_tokens, Some(3));
    assert_eq!(second.total().input_tokens, Some(5));
}
