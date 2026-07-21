//! The half of the cache that survives the process.
//!
//! dspy keeps replies on disk as well as in memory — `~/.dspy_cache` by default, 30 GB, under
//! `diskcache.FanoutCache`. That is what makes a second run of a compile cost nothing where the
//! first paid for every call, which is the difference between iterating on an optimizer and
//! waiting for one.
//!
//! Entries are JSON rather than upstream's pickle. Nothing else reads this directory, so the
//! format is ours to choose, and pickle would mean either a Python dependency or emulating its
//! bytes; JSON also means a corrupt entry is readable when working out why it was corrupt. The
//! directory is `~/.dsrs_cache` rather than dspy's, because the same path holding two
//! incompatible formats would be worse than a second directory.

use std::fs;
use std::path::{Path, PathBuf};

use crate::lm::LmResponse;

/// dspy's `DISK_CACHE_LIMIT`, 30 GB.
const DEFAULT_SIZE_LIMIT: u64 = 30_000_000_000;

/// The variable naming the directory, upstream's `DSPY_CACHEDIR` under a name that cannot
/// collide with an actual dspy install's cache.
const CACHE_DIR_VAR: &str = "DSRS_CACHEDIR";

/// The variable naming the byte budget, upstream's `DSPY_CACHE_LIMIT`.
const SIZE_LIMIT_VAR: &str = "DSRS_CACHE_LIMIT";

/// Replies kept as files, one per request.
pub struct DiskCache {
    root: PathBuf,
    size_limit: u64,
}

impl DiskCache {
    pub fn new(root: impl Into<PathBuf>, size_limit: u64) -> Self {
        Self {
            root: root.into(),
            size_limit,
        }
    }

    /// The directory and budget the environment names, or `~/.dsrs_cache` at 30 GB.
    ///
    /// `None` when there is no home directory to fall back to and nothing named one, which is
    /// the case in a sandbox with no `HOME`. A cache is an optimisation, so its absence is not
    /// an error a caller should have to handle.
    pub fn from_env() -> Option<Self> {
        let root = match std::env::var_os(CACHE_DIR_VAR) {
            Some(named) => PathBuf::from(named),
            None => PathBuf::from(std::env::var_os("HOME")?).join(".dsrs_cache"),
        };
        let size_limit = std::env::var(SIZE_LIMIT_VAR)
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(DEFAULT_SIZE_LIMIT);
        Some(Self::new(root, size_limit))
    }

    /// Two characters of the key name a subdirectory, so a long-lived cache is not one directory
    /// holding a million files — which is where several filesystems stop being quick about it.
    /// Upstream shards for the same reason, calling it a fanout.
    fn path(&self, key: &str) -> PathBuf {
        let (shard, rest) = key.split_at(2.min(key.len()));
        self.root.join(shard).join(format!("{rest}.json"))
    }

    /// The reply kept under this key, if one is there and still readable.
    ///
    /// An entry that will not parse is removed rather than reported: it means a half-written file
    /// or a format that has moved on, and either way the call it stands for can simply be made
    /// again. Upstream deletes an undeserialisable entry for the same reason.
    pub fn get(&self, key: &str) -> Option<LmResponse> {
        let path = self.path(key);
        let raw = fs::read(&path).ok()?;
        match serde_json::from_slice(&raw) {
            Ok(response) => Some(response),
            Err(error) => {
                tracing::debug!(%error, path = %path.display(), "discarding an unreadable entry");
                let _ = fs::remove_file(&path);
                None
            }
        }
    }

    /// Keep this reply, pruning first if the directory has outgrown its budget.
    ///
    /// Failing to write is not an error the caller hears about. The reply is already in hand and
    /// already in memory; a full disk or a read-only home should cost the next run its warm start
    /// rather than cost this run its answer.
    pub fn put(&self, key: &str, response: &LmResponse) {
        let path = self.path(key);
        let Some(parent) = path.parent() else { return };
        if let Err(error) = fs::create_dir_all(parent) {
            tracing::debug!(%error, "could not make room for a cache entry");
            return;
        }
        let Ok(encoded) = serde_json::to_vec(response) else {
            return;
        };
        if let Err(error) = fs::write(&path, &encoded) {
            tracing::debug!(%error, path = %path.display(), "could not keep a cache entry");
            return;
        }
        self.prune();
    }

    /// Drop the least recently written entries until the directory is inside its budget.
    ///
    /// Modification time stands in for recency because reading a file does not reliably update
    /// its access time — `noatime` and `relatime` are both common — so a read-based ordering
    /// would be wrong on most machines rather than a few.
    fn prune(&self) {
        let mut entries = self.entries();
        let mut total: u64 = entries.iter().map(|(_, size, _)| size).sum();
        if total <= self.size_limit {
            return;
        }
        entries.sort_by_key(|(_, _, modified)| *modified);
        for (path, size, _) in entries {
            if total <= self.size_limit {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(size);
            }
        }
    }

    /// Every entry, with what it costs and when it was written.
    fn entries(&self) -> Vec<(PathBuf, u64, std::time::SystemTime)> {
        let mut found = Vec::new();
        let Ok(shards) = fs::read_dir(&self.root) else {
            return found;
        };
        for shard in shards.flatten() {
            let Ok(files) = fs::read_dir(shard.path()) else {
                continue;
            };
            for file in files.flatten() {
                let Ok(metadata) = file.metadata() else {
                    continue;
                };
                let modified = metadata.modified().unwrap_or(std::time::UNIX_EPOCH);
                found.push((file.path(), metadata.len(), modified));
            }
        }
        found
    }

    /// What the kept entries occupy, for a caller deciding whether to clear.
    pub fn size(&self) -> u64 {
        self.entries().iter().map(|(_, size, _)| size).sum()
    }

    pub fn len(&self) -> usize {
        self.entries().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget everything on disk.
    pub fn clear(&self) {
        let _ = fs::remove_dir_all(&self.root);
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::LmUsage;

    /// A directory of this test's own, removed when the test ends however it ends.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("dsrs-cache-{name}"));
            let _ = fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn reply(text: &str) -> LmResponse {
        LmResponse::text(text).with_usage(Some(LmUsage {
            input_tokens: 3,
            output_tokens: 4,
        }))
    }

    #[test]
    fn a_reply_written_is_a_reply_read_back() {
        let scratch = Scratch::new("roundtrip");
        let cache = DiskCache::new(&scratch.0, DEFAULT_SIZE_LIMIT);
        cache.put("abcdef", &reply("the reply"));

        let found = cache.get("abcdef").expect("the entry is there");
        assert_eq!(found.text_ref(), "the reply");
        assert_eq!(found.usage.expect("usage survived").input_tokens, 3);
    }

    #[test]
    fn a_key_that_was_never_written_is_a_miss() {
        let scratch = Scratch::new("miss");
        let cache = DiskCache::new(&scratch.0, DEFAULT_SIZE_LIMIT);
        assert_eq!(cache.get("nothing-here"), None);
    }

    /// The key shards into a subdirectory, so a cache with a million entries is not one
    /// directory with a million files in it.
    #[test]
    fn entries_are_sharded_by_the_first_two_characters() {
        let scratch = Scratch::new("shard");
        let cache = DiskCache::new(&scratch.0, DEFAULT_SIZE_LIMIT);
        cache.put("abcdef", &reply("x"));
        assert!(scratch.0.join("ab").join("cdef.json").exists());
    }

    /// A half-written or outdated entry must not fail the call it stands for — that call can
    /// simply be made again.
    #[test]
    fn an_unreadable_entry_is_a_miss_and_is_cleared_away() {
        let scratch = Scratch::new("corrupt");
        let cache = DiskCache::new(&scratch.0, DEFAULT_SIZE_LIMIT);
        cache.put("abcdef", &reply("the reply"));

        let path = scratch.0.join("ab").join("cdef.json");
        fs::write(&path, b"{ this is not the format").expect("the file is writable");

        assert_eq!(cache.get("abcdef"), None, "unreadable reads as absent");
        assert!(!path.exists(), "and is not left to be re-read forever");
    }

    /// An unbounded directory is the disk equivalent of a leak, and 30 GB of stale replies is
    /// not a thing to leave on someone's machine.
    #[test]
    fn the_oldest_entries_are_pruned_once_the_budget_is_gone() {
        let scratch = Scratch::new("prune");
        // Small enough that a couple of entries exceed it, so the prune has to actually run.
        let cache = DiskCache::new(&scratch.0, 200);

        cache.put("aa0000", &reply("first"));
        // mtime has whole-second resolution on some filesystems, so without this the sort has
        // no order to find and the test would pass or fail on how fast the machine is.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cache.put("bb1111", &reply("second"));
        cache.put("cc2222", &reply("third"));

        assert!(cache.size() <= 200, "pruned back inside the budget");
        assert_eq!(cache.get("aa0000"), None, "the oldest entry went first");
        assert!(cache.get("cc2222").is_some(), "the newest survived");
    }

    #[test]
    fn clearing_removes_the_directory() {
        let scratch = Scratch::new("clear");
        let cache = DiskCache::new(&scratch.0, DEFAULT_SIZE_LIMIT);
        cache.put("abcdef", &reply("the reply"));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.get("abcdef"), None);
    }

    /// A named directory wins over the default, which is how a test or a CI run keeps its cache
    /// out of the home directory.
    #[test]
    fn the_environment_names_the_directory_and_the_budget() {
        // SAFETY: single-threaded read of what this test itself just set.
        unsafe {
            std::env::set_var(CACHE_DIR_VAR, "/tmp/dsrs-cache-from-env");
            std::env::set_var(SIZE_LIMIT_VAR, "4096");
        }
        let cache = DiskCache::from_env().expect("a directory was named");
        assert_eq!(cache.root(), Path::new("/tmp/dsrs-cache-from-env"));
        assert_eq!(cache.size_limit, 4096);
        unsafe {
            std::env::remove_var(CACHE_DIR_VAR);
            std::env::remove_var(SIZE_LIMIT_VAR);
        }
    }
}
