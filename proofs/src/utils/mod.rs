//! The utils module contains small, reusable functions

pub mod arithmetic;
pub mod helpers;
pub mod rational;

pub use helpers::SerdeFormat;

/// A hash map with a fast, non-randomised hasher, for the maps that circuit
/// synthesis fills and reads once per assigned cell.
pub type FxMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
/// A hash set with a fast, non-randomised hasher. See [FxMap].
pub type FxSet<K> = std::collections::HashSet<K, rustc_hash::FxBuildHasher>;
