//! What dspy reads off a failed response's headers: `_exception_retry_after` and
//! `_exception_request_id`.
//!
//! Upstream digs these out of a litellm exception, which is where they end up after the HTTP client
//! has finished with them (`exc.response.headers`, else `exc.headers`). Here they are read from the
//! response itself, which is the same two values from a shorter path.
//!
//! Both are load-bearing rather than decorative. `retry-after` is what the retry in
//! [`retry`](crate::lm::retry) waits for instead of guessing, and a request id is what a caller
//! quotes to a provider when asking why a call failed.

use reqwest::header::HeaderMap;

/// The four names dspy tries, in its order — a request id is spelled differently by every vendor.
const REQUEST_ID_HEADERS: [&str; 4] = [
    "x-request-id",
    "request-id",
    "x-amzn-requestid",
    "x-ms-request-id",
];

/// dspy's `_exception_retry_after`: seconds the provider asked the caller to wait.
///
/// A value that will not parse as a number is no value, as upstream's `except (TypeError, ValueError)`
/// has it. That includes the HTTP-date form the spec also allows, which upstream does not read either.
pub fn retry_after(headers: &HeaderMap) -> Option<f64> {
    header(headers, "retry-after")?.parse().ok()
}

/// dspy's `_exception_request_id`: the first of the four vendor spellings that is present.
pub fn request_id(headers: &HeaderMap) -> Option<String> {
    REQUEST_ID_HEADERS
        .into_iter()
        .find_map(|name| header(headers, name))
        .map(str::to_owned)
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
}

#[cfg(test)]
mod tests {
    /// The committed table, generated from CPython and dspy by
    /// `scripts/generate_constants_fixture.py`.
    fn tables() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/constants/tables.json");
        let text = std::fs::read_to_string(&path).expect("the constants golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    /// dspy tries the four in order and takes the first present, so the order decides which id a
    /// caller is shown when a provider sends more than one. Read out of `_exception_request_id`'s
    /// own AST rather than copied.
    #[test]
    fn the_request_id_headers_are_dspys_in_dspys_order() {
        let tables = tables();
        let recorded: Vec<&str> = tables["request_id_headers"]
            .as_array()
            .expect("headers")
            .iter()
            .map(|name| name.as_str().expect("a header"))
            .collect();
        assert_eq!(REQUEST_ID_HEADERS.as_slice(), recorded.as_slice());
    }

    use super::*;

    fn with(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).expect("a header name"),
                value.parse().expect("a header value"),
            );
        }
        headers
    }

    #[test]
    fn a_numeric_retry_after_is_read_as_seconds() {
        assert_eq!(retry_after(&with(&[("retry-after", "30")])), Some(30.0));
        assert_eq!(retry_after(&with(&[("retry-after", "0.25")])), Some(0.25));
        assert_eq!(retry_after(&with(&[])), None);
    }

    /// The HTTP-date form is also legal and upstream's `float()` cannot read it either, so it reads
    /// as absent rather than as zero — which would turn a one-minute wait into no wait at all.
    #[test]
    fn a_date_shaped_retry_after_is_no_value() {
        assert_eq!(
            retry_after(&with(&[("retry-after", "Wed, 21 Oct 2015 07:28:00 GMT")])),
            None
        );
    }

    /// The four spellings, and dspy's precedence between them.
    #[test]
    fn the_first_vendor_spelling_present_is_the_request_id() {
        assert_eq!(
            request_id(&with(&[("x-request-id", "abc")])).as_deref(),
            Some("abc")
        );
        assert_eq!(
            request_id(&with(&[("x-ms-request-id", "azure")])).as_deref(),
            Some("azure")
        );
        assert_eq!(
            request_id(&with(&[
                ("x-request-id", "first"),
                ("request-id", "second")
            ]))
            .as_deref(),
            Some("first"),
            "upstream tries x-request-id first"
        );
        assert_eq!(request_id(&with(&[("x-trace-id", "no")])), None);
    }
}
