//! dspy `utils/exceptions.py`: which kind of failure this was, and what a caller may do about it.
//!
//! Upstream states it as fourteen exception classes in a tree. The tree carries two things a caller
//! acts on — the leaf's identity, and whether it descends from `LMProviderError` (the provider
//! answered) or from `LMConfigurationError` (it never got that far) — so this is an enum plus the
//! two ancestry questions, rather than fourteen Rust types nobody can match on exhaustively.
//!
//! The codes are upstream's `default_code`, verbatim, because they are the stable name a caller
//! logs or compares against.

use std::fmt;

/// Which kind of LM failure this is. dspy's `LMError` subclasses, flattened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LmErrorKind {
    /// The request never reached the provider: DNS, TLS, a reset connection.
    Transport,
    /// The LM or its client is set up wrong.
    Configuration,
    /// Credentials or provider settings are missing outright.
    NotConfigured,
    /// The model or provider cannot do what was asked of it.
    UnsupportedFeature,
    /// The provider answered, and the answer was a failure this does not name more precisely.
    Provider,
    /// A failure with no status and no recognisable shape.
    Unexpected,
    /// Rejected credentials. 401 or 403.
    Auth,
    /// Out of credit or over a budget. 402.
    Billing,
    /// Too many requests. 429.
    RateLimit,
    /// The provider read the request and refused it. 4xx.
    InvalidRequest,
    /// No such model. 404.
    UnsupportedModel,
    /// The provider took too long. 408.
    Timeout,
    /// The provider broke. 5xx.
    Server,
}

impl LmErrorKind {
    /// dspy's `default_code`: the stable name for this kind.
    pub fn code(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::Configuration => "configuration",
            Self::NotConfigured => "not_configured",
            Self::UnsupportedFeature => "unsupported_feature",
            Self::Provider => "provider",
            Self::Unexpected => "unexpected",
            Self::Auth => "auth",
            Self::Billing => "billing",
            Self::RateLimit => "rate_limit",
            Self::InvalidRequest => "invalid_request",
            Self::UnsupportedModel => "unsupported_model",
            Self::Timeout => "timeout",
            Self::Server => "server",
        }
    }

    /// dspy's `is_retryable_lm_error`: whether asking again is generally safe.
    ///
    /// Advisory, as upstream says — a caller should still honour
    /// [`retry_after`](super::LmFailure::retry_after) when the provider sent one. The four kinds
    /// are upstream's `_RETRYABLE_LM_ERRORS` and no more: a rejected key or a malformed request
    /// fails again identically, and retrying it only spends the caller's time.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::RateLimit | Self::Timeout | Self::Server | Self::Transport
        )
    }

    /// Whether the provider answered at all — dspy's `LMProviderError` subtree.
    ///
    /// The distinction a caller needs before blaming their own setup: everything below it means a
    /// request was made and refused, and everything else means it never got that far.
    pub fn is_from_provider(self) -> bool {
        matches!(
            self,
            Self::Provider
                | Self::Auth
                | Self::Billing
                | Self::RateLimit
                | Self::InvalidRequest
                | Self::UnsupportedModel
                | Self::Timeout
                | Self::Server
        )
    }

    /// Whether this is the caller's own setup — dspy's `LMConfigurationError` subtree.
    pub fn is_configuration(self) -> bool {
        matches!(self, Self::Configuration | Self::NotConfigured)
    }

    /// dspy's `_lm_error_class_from_status`: an HTTP status, as a kind.
    ///
    /// The order matters where the ranges overlap — 402, 404, 408 and 429 are all 4xx, and each is
    /// named before the range that would otherwise swallow it.
    pub fn from_status(status: Option<u16>) -> Self {
        let Some(status) = status else {
            return Self::Unexpected;
        };
        match status {
            401 | 403 => Self::Auth,
            402 => Self::Billing,
            404 => Self::UnsupportedModel,
            408 => Self::Timeout,
            429 => Self::RateLimit,
            400..=499 => Self::InvalidRequest,
            500.. => Self::Server,
            _ => Self::Provider,
        }
    }
}

impl fmt::Display for LmErrorKind {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(self.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The status map, at every boundary upstream names. A 402 read as a plain 4xx would tell a
    /// caller their request was malformed when their card was declined.
    #[test]
    fn a_status_becomes_the_kind_dspy_gives_it() {
        for (status, expected) in [
            (401, LmErrorKind::Auth),
            (403, LmErrorKind::Auth),
            (402, LmErrorKind::Billing),
            (404, LmErrorKind::UnsupportedModel),
            (408, LmErrorKind::Timeout),
            (429, LmErrorKind::RateLimit),
            (400, LmErrorKind::InvalidRequest),
            (422, LmErrorKind::InvalidRequest),
            (499, LmErrorKind::InvalidRequest),
            (500, LmErrorKind::Server),
            (503, LmErrorKind::Server),
        ] {
            assert_eq!(LmErrorKind::from_status(Some(status)), expected, "{status}");
        }
    }

    /// No status at all is `Unexpected`, not `Provider` — nothing was heard back to attribute.
    #[test]
    fn no_status_is_unexpected() {
        assert_eq!(LmErrorKind::from_status(None), LmErrorKind::Unexpected);
    }

    /// Exactly upstream's `_RETRYABLE_LM_ERRORS`, and nothing else. Retrying a rejected key just
    /// spends the caller's time, and retrying a malformed request fails the same way twice.
    #[test]
    fn only_the_four_retryable_kinds_are_retryable() {
        let retryable: Vec<&str> = [
            LmErrorKind::Transport,
            LmErrorKind::Configuration,
            LmErrorKind::NotConfigured,
            LmErrorKind::UnsupportedFeature,
            LmErrorKind::Provider,
            LmErrorKind::Unexpected,
            LmErrorKind::Auth,
            LmErrorKind::Billing,
            LmErrorKind::RateLimit,
            LmErrorKind::InvalidRequest,
            LmErrorKind::UnsupportedModel,
            LmErrorKind::Timeout,
            LmErrorKind::Server,
        ]
        .into_iter()
        .filter(|kind| kind.is_retryable())
        .map(LmErrorKind::code)
        .collect();
        assert_eq!(
            retryable,
            vec!["transport", "rate_limit", "timeout", "server"]
        );
    }

    /// The two ancestry questions are what the tree is for, and they do not overlap.
    #[test]
    fn provider_and_configuration_are_disjoint() {
        for kind in [
            LmErrorKind::Auth,
            LmErrorKind::Server,
            LmErrorKind::RateLimit,
        ] {
            assert!(kind.is_from_provider(), "{kind}");
            assert!(!kind.is_configuration(), "{kind}");
        }
        for kind in [LmErrorKind::Configuration, LmErrorKind::NotConfigured] {
            assert!(kind.is_configuration(), "{kind}");
            assert!(!kind.is_from_provider(), "{kind}");
        }
        // Transport sits under neither: the request left, and nothing answered it.
        assert!(!LmErrorKind::Transport.is_from_provider());
        assert!(!LmErrorKind::Transport.is_configuration());
    }
}
