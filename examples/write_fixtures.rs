//! Write the synthetic reference pairs of `tests/reference.rs` as 32-bit float WAVs, so the
//! Python 2f tool can be run on exactly the same samples (its outputs become the expected
//! values of the test).
//!
//! ```text
//! cargo run -p sori-metrics --example write_fixtures -- <out_dir>
//! ```
//! Writes `<name>.ref.wav` and `<name>.test.wav` per pair and `pairs.tsv`, and for the
//! source-separation sets of `tests/separation.rs` `<name>.ref<j>.wav` / `<name>.est<j>.wav`
//! per source.

use std::fmt::Write as _;

#[path = "../tests/common/separation.rs"]
mod separation;
#[path = "../tests/common/signals.rs"]
mod signals;

fn main() {
    let out = std::path::PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&out).expect("output directory");
    let mut list = String::new();
    for p in signals::all() {
        let r = out.join(format!("{}.ref.wav", p.name));
        let t = out.join(format!("{}.test.wav", p.name));
        std::fs::write(
            &r,
            sori_metrics::wav::encode_f32(48_000, p.channels, &p.reference),
        )
        .expect("write reference");
        std::fs::write(
            &t,
            sori_metrics::wav::encode_f32(48_000, p.channels, &p.test),
        )
        .expect("write test");
        let _ = writeln!(list, "{}\t{}", r.display(), t.display());
    }
    std::fs::write(out.join("pairs.tsv"), list).expect("write pairs.tsv");
    for set in separation::all() {
        for (j, (r, e)) in set.references.iter().zip(&set.estimates).enumerate() {
            for (kind, x) in [("ref", r), ("est", e)] {
                let path = out.join(format!("{}.{kind}{j}.wav", set.name));
                std::fs::write(
                    path,
                    sori_metrics::wav::encode_f32(set.rate, set.channels, x),
                )
                .expect("write separation fixture");
            }
        }
    }
}
