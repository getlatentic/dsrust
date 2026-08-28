//! The protocol-4 opcodes `pickle.dumps` writes, as `pickletools` names them.
//!
//! A table of its own because it is a transcription: every byte here is CPython's, and reading it
//! against the pickle module should not mean skipping past a pickler to do so.

pub(super) const PROTO: u8 = 0x80;
pub(super) const FRAME: u8 = 0x95;
pub(super) const MARK: u8 = b'(';
pub(super) const STOP: u8 = b'.';
pub(super) const NONE: u8 = b'N';
pub(super) const NEWTRUE: u8 = 0x88;
pub(super) const NEWFALSE: u8 = 0x89;
pub(super) const BININT: u8 = b'J';
pub(super) const BININT1: u8 = b'K';
pub(super) const BININT2: u8 = b'M';
pub(super) const LONG1: u8 = 0x8a;
pub(super) const BINFLOAT: u8 = b'G';
pub(super) const SHORT_BINUNICODE: u8 = 0x8c;
pub(super) const BINUNICODE: u8 = b'X';
pub(super) const BINUNICODE8: u8 = 0x8d;
pub(super) const EMPTY_DICT: u8 = b'}';
pub(super) const SETITEM: u8 = b's';
pub(super) const SETITEMS: u8 = b'u';
pub(super) const EMPTY_LIST: u8 = b']';
pub(super) const APPEND: u8 = b'a';
pub(super) const APPENDS: u8 = b'e';
pub(super) const EMPTY_TUPLE: u8 = b')';
pub(super) const TUPLE: u8 = b't';
pub(super) const TUPLE1: u8 = 0x85;
pub(super) const TUPLE2: u8 = 0x86;
pub(super) const TUPLE3: u8 = 0x87;
pub(super) const NEWOBJ: u8 = 0x81;
pub(super) const BUILD: u8 = b'b';
pub(super) const STACK_GLOBAL: u8 = 0x93;
pub(super) const MEMOIZE: u8 = 0x94;
pub(super) const BINGET: u8 = b'h';
pub(super) const LONG_BINGET: u8 = b'j';
pub(super) const PROTOCOL: u8 = 4;
