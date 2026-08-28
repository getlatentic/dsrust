//! Protocol 4's framing, which decides where a `FRAME` opcode falls and what escapes one.
//!
//! CPython buffers what it writes and flushes it as a length-prefixed frame once the buffer
//! reaches 64 KiB, checking at every object boundary. A payload at or above that size skips the
//! buffer entirely, forcing the pending frame closed early — which is why a demo holding one long
//! string produces three pieces rather than one.

use super::opcodes::FRAME;

/// `pickle._Framer._FRAME_SIZE_TARGET`, and the minimum below which no `FRAME` opcode is written.
const FRAME_TARGET: usize = 64 * 1024;
const FRAME_MIN: usize = 4;

#[derive(Default)]
pub(super) struct Framer {
    out: Vec<u8>,
    pending: Vec<u8>,
}

impl Framer {
    /// `_Framer.commit_frame`: flush the pending frame once it reaches the target, prefixing its
    /// length unless the frame is too short to be worth an opcode.
    pub(super) fn commit(&mut self, force: bool) {
        if self.pending.is_empty() || !(force || self.pending.len() >= FRAME_TARGET) {
            return;
        }
        if self.pending.len() >= FRAME_MIN {
            self.out.push(FRAME);
            self.out
                .extend_from_slice(&(self.pending.len() as u64).to_le_bytes());
        }
        self.out.append(&mut self.pending);
    }

    /// The check `Pickler.save` opens with, before it does anything else — including before a memo
    /// hit, which is why a back-reference can start a frame.
    pub(super) fn at_an_object(&mut self) {
        self.commit(false);
    }

    pub(super) fn write(&mut self, data: &[u8]) {
        self.pending.extend_from_slice(data);
    }

    pub(super) fn push(&mut self, byte: u8) {
        self.pending.push(byte);
    }

    /// Written before any frame exists, which is where `PROTO` goes.
    pub(super) fn unframed(&mut self, data: &[u8]) {
        self.out.extend_from_slice(data);
    }

    /// `_Framer.write_large_bytes`: a payload at or above the frame target closes the pending
    /// frame and is written outside one, so neither header nor payload is ever buffered.
    pub(super) fn large(&mut self, header: &[u8], payload: &[u8]) {
        self.commit(true);
        self.out.extend_from_slice(header);
        self.out.extend_from_slice(payload);
    }

    /// Whether a payload of this size takes the unbuffered path.
    pub(super) fn is_large(size: usize) -> bool {
        size >= FRAME_TARGET
    }

    pub(super) fn finish(mut self) -> Vec<u8> {
        self.commit(true);
        self.out
    }
}
