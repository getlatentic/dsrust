//! CPython `pickle.dumps` at protocol 4, for the values a dspy demo can hold.
//!
//! Not a general pickler: it writes exactly the shapes `Hasher.hash(tuple(demos))` reaches, which
//! is a tuple of `dspy.primitives.example.Example` whose `_store` holds JSON. That is enough
//! because the bytes are not the point — their **digest** is, and it seeds the generator
//! `BootstrapFewShot` picks a demo with. A pickler that agreed about everything except the memo
//! table would produce a different seed and therefore a different demo.
//!
//! Three parts of the format decide the bytes and none of them are guessable from the value:
//!
//! * **The memo is keyed by object identity, not by equality.** Python emits a back-reference for
//!   the *same object* and writes two equal strings twice. Which is which follows from where the
//!   string came from, and the split is the demo's own: see [`Pickler::shared`].
//! * **Framing.** Protocol 4 buffers into 64 KiB frames, and a payload at or above that size is
//!   written outside the frame entirely, forcing the pending one closed early.
//! * **Which opcode a size selects**, for strings and for integers, at each of its boundaries.
//!
//! Held byte-for-byte against `optimize/hasher.json`, which records `pickle.dumps` itself.

use std::collections::HashMap;

use serde_json::Value;

use framing::Framer;
use opcodes::*;

use crate::Example;

mod framing;
mod opcodes;

/// `pickle.Pickler._BATCHSIZE`: how many items a single `SETITEMS`/`APPENDS` run covers.
const BATCH: usize = 1000;

/// `dspy.primitives.example.Example`, as `save_global` spells it from `__module__`/`__qualname__`.
const EXAMPLE_MODULE: &str = "dspy.primitives.example";
const EXAMPLE_QUALNAME: &str = "Example";

/// dspy `Example(augmented=True, ...)`: the key marking a demo the teacher earned.
const AUGMENTED: &str = "augmented";

/// Which half of a demo a name or value came from, which is what decides whether CPython holds
/// one object for it. The caller's inputs repeat between demos; the parsed outputs never do.
#[derive(Clone, Copy)]
enum Side {
    Input,
    Output,
}

/// Whether CPython holds exactly one object for this string regardless of how it was built: the
/// empty string, and the 256 one-character latin-1 strings, are interpreter-wide singletons.
fn cached_by_the_interpreter(text: &str) -> bool {
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (None, _) => true,
        (Some(only), None) => (only as u32) < 256,
        _ => false,
    }
}

/// Memo keys. A string and a container that render the same text are different objects, so the
/// two namespaces are kept apart; a string's key is its own bytes, so an input value equal to an
/// input name shares with it exactly as CPython's interning does.
fn string_key(text: &str) -> String {
    format!("s{text}")
}

fn container_key(value: &Value) -> String {
    format!("j{value}")
}

/// `pickle.dumps(tuple(demos))`.
pub(super) fn demos(demos: &[Example]) -> Vec<u8> {
    let mut pickler = Pickler::default();
    pickler.framer.unframed(&[PROTO, PROTOCOL]);
    pickler.tuple_of_examples(demos);
    pickler.framer.push(STOP);
    pickler.framer.finish()
}

#[derive(Default)]
struct Pickler {
    framer: Framer,
    /// The next `MEMOIZE` index. Python's memo is a dict keyed by `id`; only its *size* is ever
    /// written, so a counter says the same thing.
    memo: usize,
    /// The objects CPython holds one of, and the slot each was memoized into.
    ///
    /// Which strings those are is decided by the input/output split, not by whether they are a
    /// name or a value. A demo is `Example(augmented=True, **inputs, **outputs)`:
    ///
    /// * The **inputs** are the caller's own objects. A program that calls one predictor twice
    ///   passes the same question variable to both, so the name *and* the value repeat as
    ///   back-references — `'question'` and `'what is up?'` alike.
    /// * The **outputs** were parsed out of a completion and are new objects every time. Two
    ///   equal ones are written twice, even inside a single demo.
    ///
    /// Measured rather than reasoned: `ChatAdapter`, `JSONAdapter` and `XMLAdapter` were each
    /// asked, and all three return output names and values sharing with neither the signature nor
    /// a previous parse nor each other. `optimize/hasher.json` pins both sides, and one of its cases
    /// is the tuple an actual compile passed to `Hasher.hash`, captured rather than written.
    ///
    /// The residue is equality standing in for identity: two *equal* inputs that were separate
    /// objects back-reference here and would not upstream. Nothing in a program that reaches this
    /// branch produces them — the branch needs one predictor called repeatedly, and repeated calls
    /// take their equal arguments from the same variable.
    shared: HashMap<String, usize>,
    /// The `Example` class, memoized once by `save_global` and back-referenced by every demo after.
    class: Option<usize>,
}

impl Pickler {
    fn memoize(&mut self) -> usize {
        self.framer.push(MEMOIZE);
        let slot = self.memo;
        self.memo += 1;
        slot
    }

    fn get(&mut self, slot: usize) {
        if slot < 256 {
            self.framer.write(&[BINGET, slot as u8]);
        } else {
            self.framer.push(LONG_BINGET);
            self.framer.write(&(slot as u32).to_le_bytes());
        }
    }

    /// A string Python holds one object for: written once, back-referenced after.
    fn interned(&mut self, text: &str) {
        self.framer.at_an_object();
        if let Some(&slot) = self.shared.get(&string_key(text)) {
            self.get(slot);
            return;
        }
        self.str_bytes(text);
        let slot = self.memoize();
        self.shared.insert(string_key(text), slot);
    }

    /// An input-side value: the caller's object, so an equal one later is a back-reference.
    ///
    /// A container takes part whole, as Python's memo does — a passage list passed to both hops is
    /// one `BINGET`, not a rewritten list. Its first appearance still writes its contents through
    /// this same path, so a string inside it can be back-referenced on its own afterwards.
    fn shared_value(&mut self, value: &Value) {
        self.framer.at_an_object();
        match value {
            Value::String(text) => self.interned(text),
            Value::Array(_) | Value::Object(_) => {
                let key = container_key(value);
                if let Some(&slot) = self.shared.get(&key) {
                    self.get(slot);
                    return;
                }
                let slot = self.container(value, Side::Input);
                self.shared.insert(key, slot);
            }
            _ => self.value(value),
        }
    }

    /// One `name: value` pair of a `_store`, written by the side it came from.
    fn field(&mut self, side: Side, name: &str, value: &Value) {
        match side {
            Side::Input => {
                self.interned(name);
                self.shared_value(value);
            }
            Side::Output => {
                self.fresh_str(name);
                self.value(value);
            }
        }
    }

    /// A string that is its own object every time it appears, as a parsed one is — unless CPython
    /// keeps only one of it whatever built it.
    ///
    /// The interpreter caches the empty string and every single latin-1 character, so a one-letter
    /// output field or a one-letter answer is the *same* object across demos and back-references
    /// like an input would. A single character above U+00FF is not cached and stays fresh.
    fn fresh_str(&mut self, text: &str) {
        if cached_by_the_interpreter(text) {
            self.interned(text);
            return;
        }
        self.framer.at_an_object();
        self.str_bytes(text);
        self.memoize();
    }

    /// `save_str`: the opcode a length selects, and the large-payload path at the frame target.
    fn str_bytes(&mut self, text: &str) {
        let encoded = text.as_bytes();
        let n = encoded.len();
        if n <= 0xff {
            self.framer.write(&[SHORT_BINUNICODE, n as u8]);
            self.framer.write(encoded);
        } else if n > 0xffff_ffff {
            let mut header = vec![BINUNICODE8];
            header.extend_from_slice(&(n as u64).to_le_bytes());
            self.framer.large(&header, encoded);
        } else if Framer::is_large(n) {
            let mut header = vec![BINUNICODE];
            header.extend_from_slice(&(n as u32).to_le_bytes());
            self.framer.large(&header, encoded);
        } else {
            self.framer.write(&[BINUNICODE]);
            self.framer.write(&(n as u32).to_le_bytes());
            self.framer.write(encoded);
        }
    }

    /// `save_tuple` over the demos, then `STOP`'s caller memoizes it.
    fn tuple_of_examples(&mut self, demos: &[Example]) {
        self.framer.at_an_object();
        if demos.is_empty() {
            self.framer.push(EMPTY_TUPLE);
            return;
        }
        if demos.len() > 3 {
            self.framer.push(MARK);
        }
        for demo in demos {
            self.example(demo);
        }
        self.framer.push(match demos.len() {
            1 => TUPLE1,
            2 => TUPLE2,
            3 => TUPLE3,
            _ => TUPLE,
        });
        self.memoize();
    }

    /// `save_reduce` for an object whose `__reduce_ex__` answers `copyreg.__newobj__`: the class,
    /// an empty argument tuple, `NEWOBJ`, then the instance `__dict__` and `BUILD`.
    fn example(&mut self, demo: &Example) {
        self.framer.at_an_object();
        self.class_object();
        self.framer.at_an_object();
        self.framer.push(EMPTY_TUPLE);
        self.framer.push(NEWOBJ);
        self.memoize();
        self.state(demo);
        self.framer.push(BUILD);
    }

    /// `save_type` → `save_global`: the module and qualified name as two strings, joined on the
    /// stack. The class is one object, so every demo after the first back-references it.
    fn class_object(&mut self) {
        if let Some(slot) = self.class {
            self.framer.at_an_object();
            self.get(slot);
            return;
        }
        self.interned(EXAMPLE_MODULE);
        self.interned(EXAMPLE_QUALNAME);
        self.framer.push(STACK_GLOBAL);
        self.class = Some(self.memoize());
    }

    /// The instance `__dict__`: `_store`, `_demos` and `_input_keys`, in the order
    /// `Example.__init__` assigns them.
    ///
    /// `_input_keys` is `None` here rather than a set, because the demo `BootstrapFewShot` builds
    /// is `Example(augmented=True, **inputs, **outputs)` with no `with_inputs` after it.
    fn state(&mut self, demo: &Example) {
        self.framer.at_an_object();
        self.framer.push(EMPTY_DICT);
        self.memoize();
        self.framer.push(MARK);
        self.interned("_store");
        self.store(demo);
        self.interned("_demos");
        self.framer.at_an_object();
        self.framer.push(EMPTY_LIST);
        self.memoize();
        self.interned("_input_keys");
        self.framer.at_an_object();
        self.framer.push(NONE);
        self.framer.push(SETITEMS);
    }

    fn store(&mut self, demo: &Example) {
        self.framer.at_an_object();
        self.framer.push(EMPTY_DICT);
        self.memoize();
        let fields: Vec<(&str, &Value)> = demo.fields().collect();
        self.batched(&fields, SETITEM, SETITEMS, |this, (name, value)| {
            // `augmented` is a literal in `_bootstrap_one_example`, so CPython interned it and
            // every demo after the first names the same object.
            let side = match *name == AUGMENTED || demo.is_input(name) {
                true => Side::Input,
                false => Side::Output,
            };
            this.field(side, name, value);
        });
    }

    /// `_batch_setitems` and `_batch_appends`, which share a shape: runs of at most `_BATCHSIZE`,
    /// each written with `MARK ... APPENDS` unless the run holds exactly one item.
    ///
    /// A trailing full batch is followed by an empty one that writes nothing, which is why the
    /// loop is over chunks rather than over a `while` on the remainder.
    fn batched<T>(
        &mut self,
        items: &[T],
        single: u8,
        many: u8,
        mut each: impl FnMut(&mut Self, &T),
    ) {
        for chunk in items.chunks(BATCH) {
            if chunk.len() > 1 {
                self.framer.push(MARK);
                for item in chunk {
                    each(self, item);
                }
                self.framer.push(many);
            } else if let [item] = chunk {
                each(self, item);
                self.framer.push(single);
            }
        }
    }

    /// A JSON value as CPython pickles the Python object it deserializes to.
    fn value(&mut self, value: &Value) {
        self.framer.at_an_object();
        match value {
            Value::Null => self.framer.push(NONE),
            Value::Bool(true) => self.framer.push(NEWTRUE),
            Value::Bool(false) => self.framer.push(NEWFALSE),
            Value::Number(number) => self.number(number),
            Value::String(text) => self.fresh_str(text),
            Value::Array(_) | Value::Object(_) => {
                self.container(value, Side::Output);
            }
        }
    }

    /// A list or a dict, written with the given rule for what is inside it, and memoized as one
    /// object — which is the slot a later back-reference to the whole container names.
    fn container(&mut self, value: &Value, side: Side) -> usize {
        match value {
            Value::Array(items) => {
                self.framer.push(EMPTY_LIST);
                let slot = self.memoize();
                self.batched(items, APPEND, APPENDS, |this, item| match side {
                    Side::Input => this.shared_value(item),
                    Side::Output => this.value(item),
                });
                slot
            }
            Value::Object(map) => {
                self.framer.push(EMPTY_DICT);
                let slot = self.memoize();
                let pairs: Vec<(&str, &Value)> = map.iter().map(|(k, v)| (k.as_str(), v)).collect();
                self.batched(&pairs, SETITEM, SETITEMS, |this, (name, value)| {
                    this.field(side, name, value);
                });
                slot
            }
            _ => unreachable!("containers only"),
        }
    }

    /// `save_long` and `save_float`. Neither memoizes — CPython's dispatch for numbers does not.
    fn number(&mut self, number: &serde_json::Number) {
        if let Some(integer) = number.as_i64() {
            self.integer(i128::from(integer));
        } else if let Some(integer) = number.as_u64() {
            self.integer(i128::from(integer));
        } else {
            self.framer.push(BINFLOAT);
            let float = number.as_f64().expect("a JSON number is one of the three");
            self.framer.write(&float.to_be_bytes());
        }
    }

    fn integer(&mut self, value: i128) {
        if (0..=0xff).contains(&value) {
            self.framer.write(&[BININT1, value as u8]);
        } else if (0..=0xffff).contains(&value) {
            self.framer.push(BININT2);
            self.framer.write(&(value as u16).to_le_bytes());
        } else if (-0x8000_0000..=0x7fff_ffff).contains(&value) {
            self.framer.push(BININT);
            self.framer.write(&(value as i32).to_le_bytes());
        } else {
            // `encode_long`: the shortest two's-complement little-endian form that keeps the sign,
            // which for a `u64` at the top of the range needs a zero byte above it.
            let mut bytes = value.to_le_bytes().to_vec();
            let sign = if value < 0 { 0xff } else { 0x00 };
            while bytes.len() > 1 && bytes[bytes.len() - 1] == sign {
                let keep = bytes[bytes.len() - 2] & 0x80 != 0;
                if keep == (sign == 0xff) {
                    bytes.pop();
                } else {
                    break;
                }
            }
            self.framer.write(&[LONG1, bytes.len() as u8]);
            self.framer.write(&bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every recorded case, compared as bytes rather than as a digest.
    ///
    /// The digest alone would prove as much — two pickles differing anywhere hash apart — but it
    /// would report every bug as one opaque mismatch. A byte comparison names the opcode.
    #[test]
    fn each_recorded_pickle_is_reproduced_exactly() {
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/conformance/optimize/hasher.json"))
                .expect("the hasher golden is valid JSON");
        assert_eq!(
            golden["protocol"].as_u64(),
            Some(u64::from(PROTOCOL)),
            "this writer is protocol {PROTOCOL}; CPython's default has moved"
        );
        let cases = golden["cases"].as_array().expect("cases");
        assert!(cases.len() >= 12, "the golden lost cases: {}", cases.len());
        for case in cases {
            let name = case["name"].as_str().expect("name");
            let ours = demos(&rebuild(case));
            assert_eq!(
                hex(&ours),
                case["pickle_hex"].as_str().expect("pickle_hex"),
                "pickle.dumps disagrees for {name}"
            );
        }
    }

    /// A demo as the golden recorded it. `toDict` keeps `_store`'s order, which is the order the
    /// pickle writes and therefore the order the digest depends on.
    ///
    /// The input split comes back too. dspy's demo has none — `_input_keys` is `None`, and the
    /// writer says so — but which fields were the call's inputs is what tells this crate which
    /// strings CPython held one object for. It is the one thing this crate's `Example` knows that
    /// upstream's demo does not, and reproducing the memo table needs it.
    fn rebuild(case: &serde_json::Value) -> Vec<Example> {
        let keys: Vec<String> = case["input_keys"]
            .as_array()
            .expect("input_keys")
            .iter()
            .map(|key| key.as_str().expect("a key").to_owned())
            .collect();
        case["demos"]
            .as_array()
            .expect("demos")
            .iter()
            .map(|fields| {
                Example::new(
                    fields
                        .as_object()
                        .expect("a demo is an object")
                        .iter()
                        .map(|(name, value)| (name.clone(), value.clone())),
                )
                .with_inputs(keys.clone())
            })
            .collect()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
