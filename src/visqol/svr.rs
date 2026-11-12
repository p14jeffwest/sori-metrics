//! Support vector regression for the ViSQOL v3 similarity-to-quality mapping (stage ①).
//!
//! ViSQOL v3 (Chinen et al. 2020, §III-E) maps the per-band NSIM vector to MOS with a libsvm
//! model. The model file is data (Google ViSQOL, Apache-2.0 — see `data/visqol/README.md`);
//! this module reads the libsvm text model format and evaluates it, written from the libsvm
//! documentation of the format and of epsilon/nu-SVR prediction:
//!
//! `f(x) = Σ_i coef_i · K(sv_i, x) − rho`, with the RBF kernel `K(u, v) = exp(−γ‖u − v‖²)`.
//!
//! Feature indices are 1-based; features missing from a support vector line are zero
//! (libsvm sparse format).

use crate::Error;

/// Kernel of a model. Only the RBF kernel is used by ViSQOL; the others are rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kernel {
    Rbf { gamma: f64 },
}

/// A loaded libsvm regression model (epsilon-SVR or nu-SVR).
#[derive(Debug, Clone)]
pub struct SvrModel {
    kernel: Kernel,
    rho: f64,
    /// Dense support vectors, `dim` values each, row-major.
    sv: Vec<f64>,
    coef: Vec<f64>,
    dim: usize,
}

impl SvrModel {
    /// Parse a libsvm text model.
    ///
    /// # Errors
    /// [`Error::Model`] for anything that is not an RBF epsilon-SVR / nu-SVR model.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let bad = |m: String| Error::Model(m);
        let mut lines = text.lines();
        let (mut svm_type, mut kernel, mut gamma, mut rho, mut total) =
            (None, None, None, None, None);
        for line in lines.by_ref() {
            let line = line.trim();
            if line == "SV" {
                break;
            }
            let mut it = line.split_whitespace();
            let key = it.next().unwrap_or("");
            let val: Vec<&str> = it.collect();
            match key {
                "svm_type" => svm_type = val.first().map(|s| (*s).to_string()),
                "kernel_type" => kernel = val.first().map(|s| (*s).to_string()),
                "gamma" => gamma = val.first().and_then(|s| s.parse::<f64>().ok()),
                "rho" => rho = val.first().and_then(|s| s.parse::<f64>().ok()),
                "total_sv" => total = val.first().and_then(|s| s.parse::<usize>().ok()),
                // nr_class, label, nr_sv, probA/probB: not needed for regression.
                _ => {}
            }
        }
        match svm_type.as_deref() {
            Some("nu_svr" | "epsilon_svr") => {}
            other => return Err(bad(format!("svm_type {other:?} is not a regression model"))),
        }
        if kernel.as_deref() != Some("rbf") {
            return Err(bad(format!("kernel_type {kernel:?} is not rbf")));
        }
        let gamma = gamma.ok_or_else(|| bad("missing gamma".into()))?;
        let rho = rho.ok_or_else(|| bad("missing rho".into()))?;
        let mut rows: Vec<(f64, Vec<(usize, f64)>)> = Vec::new();
        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let c: f64 = it
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| bad(format!("bad coefficient in {line:.40}")))?;
            let mut feats = Vec::new();
            for tok in it {
                let (i, v) = tok
                    .split_once(':')
                    .ok_or_else(|| bad(format!("bad feature {tok}")))?;
                let i: usize = i.parse().map_err(|_| bad(format!("bad index {tok}")))?;
                let v: f64 = v.parse().map_err(|_| bad(format!("bad value {tok}")))?;
                if i == 0 {
                    return Err(bad("feature index 0".into()));
                }
                feats.push((i, v));
            }
            rows.push((c, feats));
        }
        if let Some(t) = total {
            if t != rows.len() {
                return Err(bad(format!("total_sv {t} but {} vectors", rows.len())));
            }
        }
        let dim = rows
            .iter()
            .flat_map(|(_, f)| f.iter().map(|&(i, _)| i))
            .max()
            .unwrap_or(0);
        let mut sv = vec![0.0; rows.len() * dim];
        let mut coef = Vec::with_capacity(rows.len());
        for (r, (c, feats)) in rows.into_iter().enumerate() {
            coef.push(c);
            for (i, v) in feats {
                sv[r * dim + i - 1] = v;
            }
        }
        Ok(Self {
            kernel: Kernel::Rbf { gamma },
            rho,
            sv,
            coef,
            dim,
        })
    }

    /// Number of features (largest feature index in the support vectors).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of support vectors.
    pub fn len(&self) -> usize {
        self.coef.len()
    }

    /// `true` when the model has no support vectors.
    pub fn is_empty(&self) -> bool {
        self.coef.is_empty()
    }

    /// Raw regression value `Σ coef_i K(sv_i, x) − rho` (no clamping).
    ///
    /// # Panics
    /// If `x.len()` differs from [`Self::dim`].
    pub fn predict(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "feature vector length");
        let Kernel::Rbf { gamma } = self.kernel;
        let mut sum = 0.0;
        for (c, row) in self.coef.iter().zip(self.sv.chunks_exact(self.dim)) {
            let d2: f64 = row.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum();
            sum += c * (-gamma * d2).exp();
        }
        sum - self.rho
    }
}

/// The ViSQOL v3 audio-mode model shipped as data (`data/visqol/libsvm_nu_svr_model.txt`).
pub const AUDIO_MODEL: &str = include_str!("../../data/visqol/libsvm_nu_svr_model.txt");

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "svm_type nu_svr\nkernel_type rbf\ngamma 0.5\nnr_class 2\ntotal_sv 2\nrho 0.25\nSV\n2 1:1 3:2\n-1 2:1\n";

    #[test]
    fn tiny_model_by_hand() {
        let m = SvrModel::parse(TINY).unwrap();
        assert_eq!((m.dim(), m.len()), (3, 2));
        let x = [1.0, 0.0, 1.0];
        // sv1 = (1,0,2): d² = 1 → 2·e^-0.5; sv2 = (0,1,0): d² = 3 → −e^-1.5.
        let want = 2.0 * (-0.5f64).exp() - (-1.5f64).exp() - 0.25;
        assert!((m.predict(&x) - want).abs() < 1e-15);
    }

    #[test]
    fn rejects_classifiers_and_other_kernels() {
        assert!(SvrModel::parse(&TINY.replace("nu_svr", "c_svc")).is_err());
        assert!(SvrModel::parse(&TINY.replace("rbf", "linear")).is_err());
        assert!(SvrModel::parse(&TINY.replace("total_sv 2", "total_sv 3")).is_err());
    }

    #[test]
    fn audio_model_loads() {
        let m = SvrModel::parse(AUDIO_MODEL).unwrap();
        assert_eq!((m.dim(), m.len()), (32, 317));
        // Perfect similarity in every band should predict a high quality.
        let top = m.predict(&[1.0; 32]);
        let low = m.predict(&[0.5; 32]);
        assert!(top > 4.0 && top > low, "top {top} low {low}");
    }
}
