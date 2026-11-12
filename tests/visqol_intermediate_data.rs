//! Shape and internal-consistency checks for the `ViSQOL` v3 audio reference outputs of task
//! T-113 (`data/visqol-intermediate/`, `visqol-python` 3.8.0 outputs only — see its README).
//!
//! These guard the data M-03 will compare against: two decodes per public item plus three
//! synthetic pairs (51 here; 49 in the public export, which leaves out the one speech item),
//! 32 bands each, and the relations
//! between the aggregate fields and the per-patch fields that hold exactly in the recorded
//! outputs (observed, not taken from any source code).

use std::collections::BTreeMap;
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/visqol-intermediate")
}

/// Reads a TSV with a header row into (header, rows).
fn read_tsv(name: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(data_dir().join(name)).expect("read tsv");
    let mut lines = text.lines();
    let head: Vec<String> = lines
        .next()
        .expect("header")
        .split('\t')
        .map(str::to_owned)
        .collect();
    let rows = lines
        .filter(|l| !l.is_empty())
        .map(|l| l.split('\t').map(str::to_owned).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for r in &rows {
        assert_eq!(r.len(), head.len(), "{name}: column count");
    }
    (head, rows)
}

fn col(head: &[String], name: &str) -> usize {
    head.iter().position(|h| h == name).expect("column")
}

fn num(s: &str) -> f64 {
    s.parse().expect("number")
}

#[test]
fn visqol_intermediate_shape_and_relations() {
    let (ph, pairs) = read_tsv("pairs.tsv");
    let (bh, bands) = read_tsv("bands.tsv");
    let (qh, patches) = read_tsv("patches.tsv");
    let (pair_col, n) = (col(&ph, "pair"), pairs.len());
    let items: std::collections::BTreeSet<&str> = pairs
        .iter()
        .map(|r| r[pair_col].split('.').next().unwrap())
        .filter(|i| !i.starts_with("syn-"))
        .collect();
    assert!(items.len() >= 23, "public items");
    assert_eq!(n, items.len() * 2 + 3, "items x 2 decodes + 3 synthetic");
    assert_eq!(bands.len(), n * 32);

    let (p_id, p_mos, p_vn, p_np) = (
        col(&ph, "pair"),
        col(&ph, "moslqo"),
        col(&ph, "vnsim"),
        col(&ph, "n_patches"),
    );
    let (b_id, b_fv) = (col(&bh, "pair"), col(&bh, "fvnsim"));
    let (q_id, q_sim) = (col(&qh, "pair"), col(&qh, "similarity"));

    let mut fv: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for r in &bands {
        fv.entry(&r[b_id]).or_default().push(num(&r[b_fv]));
    }
    let mut npatch: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &patches {
        *npatch.entry(&r[q_id]).or_default() += 1;
        let s = num(&r[q_sim]);
        assert!((-1.0..=1.0 + 1e-12).contains(&s), "patch similarity range");
    }
    for r in &pairs {
        let id = r[p_id].as_str();
        let mos = num(&r[p_mos]);
        assert!((1.0..=5.0).contains(&mos), "{id}: moslqo range");
        let f = &fv[id];
        assert_eq!(f.len(), 32, "{id}: bands");
        // vnsim is the mean of the 32 band fvnsim values (exact in the recorded outputs).
        let mean = f.iter().sum::<f64>() / 32.0;
        assert!(
            (mean - num(&r[p_vn])).abs() < 1e-12,
            "{id}: vnsim = mean(fvnsim)"
        );
        assert_eq!(
            npatch[id],
            r[p_np].parse::<usize>().unwrap(),
            "{id}: patch count"
        );
    }
    // Identical signals: every band similarity is 1.
    assert!(fv["syn-identical"].iter().all(|&v| (v - 1.0).abs() < 1e-12));
}
