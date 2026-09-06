use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use super::{Ambience, Scoped};

thread_local! {
    static SLOT: RefCell<Option<&'static str>> = const { RefCell::new(None) };
    static SEEN: RefCell<Vec<(&'static str, Option<&'static str>)>> = const { RefCell::new(Vec::new()) };
}

struct Slot;

impl Ambience for Slot {
    type State = &'static str;
    fn exchange(held: &mut Option<&'static str>) {
        SLOT.with_borrow_mut(|installed| std::mem::swap(installed, held));
    }
}

fn installed() -> Option<&'static str> {
    SLOT.with_borrow(|slot| *slot)
}

fn note(what: &'static str) {
    SEEN.with_borrow_mut(|seen| seen.push((what, installed())));
}

fn seen() -> Vec<(&'static str, Option<&'static str>)> {
    SEEN.with_borrow_mut(std::mem::take)
}

/// Notes what was installed when it was destroyed, and can be told to hang first.
struct NoteOnDrop {
    label: &'static str,
    ready: bool,
}

impl Future for NoteOnDrop {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        if self.ready {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for NoteOnDrop {
    fn drop(&mut self) {
        note(self.label);
    }
}

#[tokio::test]
async fn a_completed_future_is_destroyed_under_its_own_scope() {
    Scoped::<Slot, _>::new(
        "outer",
        Scoped::<Slot, _>::new(
            "inner",
            NoteOnDrop {
                label: "done",
                ready: true,
            },
        ),
    )
    .await;

    assert_eq!(seen(), [("done", Some("inner"))]);
    assert_eq!(installed(), None, "and nothing is left installed");
}

#[tokio::test]
async fn a_cancelled_future_is_destroyed_under_its_own_scope() {
    let mut context = Context::from_waker(std::task::Waker::noop());
    let outer = Scoped::<Slot, _>::new("outer", async {
        let mut inner = Box::pin(Scoped::<Slot, _>::new(
            "inner",
            NoteOnDrop {
                label: "cancelled",
                ready: false,
            },
        ));
        assert!(inner.as_mut().poll(&mut context).is_pending());
        drop(inner);
        note("after");
    });
    outer.await;

    assert_eq!(
        seen(),
        [("cancelled", Some("inner")), ("after", Some("outer"))]
    );
}

#[tokio::test]
async fn interleaved_scopes_never_see_each_other() {
    let watch = |label: &'static str| {
        Scoped::<Slot, _>::new(label, async move {
            tokio::task::yield_now().await;
            note(label);
            tokio::task::yield_now().await;
        })
    };
    tokio::join!(watch("first"), watch("second"));

    let mut observed = seen();
    observed.sort();
    assert_eq!(
        observed,
        [("first", Some("first")), ("second", Some("second"))]
    );
}

#[tokio::test]
async fn an_unwinding_poll_leaves_the_enclosing_scope_standing() {
    use futures_util::FutureExt;

    Scoped::<Slot, _>::new("outer", async {
        let inner = Scoped::<Slot, _>::new("inner", async { panic!("inside") });
        let caught = std::panic::AssertUnwindSafe(inner).catch_unwind().await;
        assert!(caught.is_err());
        note("after the panic");
    })
    .await;

    assert_eq!(seen(), [("after the panic", Some("outer"))]);
    assert_eq!(installed(), None);
}
