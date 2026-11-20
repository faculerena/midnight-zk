//! The utils module contains small, reusable functions

pub mod arithmetic;
pub mod helpers;
pub mod rational;

pub use helpers::SerdeFormat;

/// Instrumentation: prints the time since the first call and since the
/// previous call, and the resident memory of the process now and at its peak
/// (on Linux), when the environment variable `ZKIR_TIMING` is set. It prints
/// to standard error and changes nothing else.
pub fn tmark(label: &str) {
    use std::{sync::{Mutex, OnceLock}, time::Instant};
    static ON: OnceLock<bool> = OnceLock::new();
    static T: OnceLock<(Instant, Mutex<Instant>)> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("ZKIR_TIMING").is_some()) {
        return;
    }
    let (t0, last) = T.get_or_init(|| { let n = Instant::now(); (n, Mutex::new(n)) });
    let now = Instant::now();
    let mut l = last.lock().unwrap();
    let mem = std::fs::read_to_string("/proc/self/status")
        .map(|s| {
            let kb = |key: &str| {
                s.lines()
                    .find_map(|line| line.strip_prefix(key))
                    .and_then(|v| v.split_whitespace().next())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0)
            };
            format!(" [rss {:5} peak {:5} MB]", kb("VmRSS:") / 1024, kb("VmHWM:") / 1024)
        })
        .unwrap_or_default();
    eprintln!(
        "[T {:8.3} +{:7.3}]{} {}",
        (now - *t0).as_secs_f64(),
        (now - *l).as_secs_f64(),
        mem,
        label
    );
    *l = now;
}

/// A hash map with a fast, non-randomised hasher, for the maps that circuit
/// synthesis fills and reads once per assigned cell.
pub type FxMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
/// A hash set with a fast, non-randomised hasher. See [FxMap].
pub type FxSet<K> = std::collections::HashSet<K, rustc_hash::FxBuildHasher>;
