# ViSQOL audio-mode SVR model (data)

- `libsvm_nu_svr_model.txt`: Google ViSQOL's audio-mode similarity-to-MOS model
  (`model/libsvm_nu_svr_model.txt` in https://github.com/google/visqol, branch `master`,
  retrieved 2026-11-12). libsvm nu-SVR, RBF kernel, 32 features, 317 support vectors.
- Copyright Google LLC, **Apache License 2.0** — full text in `LICENSE-Apache-2.0`.
- SHA-256: `1e8246ed33bf36dc5c859351f7110f2cd31f98661989715c0fcf974ec48d3e2e`
- Included unchanged and used as data only. The reader and predictor (`src/visqol/svr.rs`)
  were written from the libsvm format documentation.
