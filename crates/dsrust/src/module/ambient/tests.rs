use super::*;
use crate::signature::Signature;

fn step(named: &str) -> TraceStep {
    TraceStep {
        predictor: named.into(),
        inputs: crate::Example::default(),
        outputs: super::super::StepOutputs::Answered(crate::Example::default()),
        signature: "q -> a".parse::<Signature>().expect("parses"),
    }
}

struct RecordOnDrop {
    ready: bool,
}

impl Future for RecordOnDrop {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        if self.ready {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for RecordOnDrop {
    fn drop(&mut self) {
        record(1, step("cleanup"));
    }
}

#[tokio::test]
async fn completed_future_cleanup_belongs_to_the_inner_trace() {
    let names = Arc::new(HashMap::new());
    let (inner, outer) = recording(&names, async {
        let (_, inner) = recording(&names, RecordOnDrop { ready: true }).await;
        record(2, step("outer"));
        inner
    })
    .await;

    assert_eq!(inner.len(), 1);
    assert_eq!(inner[0].predictor, "cleanup");
    assert_eq!(outer.len(), 1);
    assert_eq!(outer[0].predictor, "outer");
    assert!(!listening());
}

#[tokio::test]
async fn cancelled_future_cleanup_does_not_enter_the_outer_trace() {
    let names = Arc::new(HashMap::new());
    let (_, outer) = recording(&names, async {
        let mut inner = Box::pin(recording(&names, RecordOnDrop { ready: false }));
        let mut context = Context::from_waker(std::task::Waker::noop());
        assert!(inner.as_mut().poll(&mut context).is_pending());
        drop(inner);
        record(2, step("outer"));
    })
    .await;

    assert_eq!(outer.len(), 1);
    assert_eq!(outer[0].predictor, "outer");
    assert!(!listening());
}

#[tokio::test]
async fn panicking_inner_future_restores_the_outer_context() {
    use futures_util::FutureExt;

    let names = Arc::new(HashMap::new());
    let (_, outer) = recording(&names, async {
        let inner = recording(&names, async {
            let _cleanup = RecordOnDrop { ready: false };
            panic!("predictor failed");
        });
        assert!(
            std::panic::AssertUnwindSafe(inner)
                .catch_unwind()
                .await
                .is_err()
        );
        record(2, step("outer"));
    })
    .await;

    assert_eq!(outer.len(), 1);
    assert_eq!(outer[0].predictor, "outer");
    assert!(!listening());
}

#[test]
fn a_trace_moves_with_its_future_between_threads() {
    let names = Arc::new(HashMap::from([(1, "named".into())]));
    let mut first_poll = true;
    let work = std::future::poll_fn(move |_| {
        record(1, step("default"));
        if std::mem::take(&mut first_poll) {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    });
    let trace = Box::pin(recording(&names, work));
    let recorded = std::thread::scope(|threads| {
        let mut trace = threads
            .spawn(move || {
                let mut trace = trace;
                assert!(
                    trace
                        .as_mut()
                        .poll(&mut Context::from_waker(std::task::Waker::noop()))
                        .is_pending()
                );
                assert!(!listening());
                trace
            })
            .join()
            .unwrap();
        threads
            .spawn(move || {
                let result = trace
                    .as_mut()
                    .poll(&mut Context::from_waker(std::task::Waker::noop()));
                assert!(!listening());
                let Poll::Ready(((), recorded)) = result else {
                    panic!("second poll must finish")
                };
                recorded
            })
            .join()
            .unwrap()
    });

    assert_eq!(recorded.len(), 2);
    assert!(recorded.iter().all(|step| step.predictor == "named"));
}

#[tokio::test]
async fn two_runs_interleaved_on_one_thread_keep_their_traces_apart() {
    let names = Arc::new(HashMap::new());
    let one = recording(&names, async {
        tokio::task::yield_now().await;
        record(1, step("first"));
        tokio::task::yield_now().await;
    });
    let two = recording(&names, async {
        tokio::task::yield_now().await;
        record(2, step("second"));
        tokio::task::yield_now().await;
    });
    let ((_, first), (_, second)) = tokio::join!(one, two);

    assert_eq!(
        first
            .iter()
            .map(|s| s.predictor.as_str())
            .collect::<Vec<_>>(),
        ["first"],
        "the first run recorded only its own call"
    );
    assert_eq!(
        second
            .iter()
            .map(|s| s.predictor.as_str())
            .collect::<Vec<_>>(),
        ["second"],
    );
}

#[tokio::test]
async fn a_call_outside_any_run_is_not_recorded() {
    let names = Arc::new(HashMap::new());
    let (_, recorded) = recording(&names, async {}).await;
    assert!(recorded.is_empty());
    assert!(!listening(), "the buffer came down with the run");
    record(1, step("stray"));
}
