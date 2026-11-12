# ViSQOL v3 audio 중간 출력값 (T-113, D-0268)

M-03(Rust ViSQOL 자체 구현)이 단계마다 맞춰 볼 정답값. `visqol-python`을 **실행만** 해서 결과 객체의 공개 필드를 모두 남겼다. 필드 이름은 결과 객체의 `dir()`로 확인했고, `visqol-python`·Google ViSQOL 소스는 읽지 않았다. 음원·WAV는 커밋하지 않는다(수치만).

## 실행 설정

| 항목 | 값 |
|---|---|
| 도구 | `visqol-python` 3.8.0 (PyPI), Python 3.13.18, numpy 2.5.4, scipy 1.18.1, soundfile 0.14.0, numba 0.68.1 |
| 호출 | `VisqolApi().create(mode="audio")` → `measure_from_arrays(ref.mean(axis=1), deg.mean(axis=1), sample_rate=48000)` — `quality_metrics.visqol`과 같음 |
| `create` 인자(실행 시 `_create_kwargs`) | `mode="audio"`, `search_window=60`, `use_unscaled_speech=False`, `use_lattice_model=None`(→ 관리자 `use_lattice_model=False`), `disable_global_alignment=False`, `disable_realignment=False` |
| 모델 | 패키지 안 `visqol/model/libsvm_nu_svr_model.txt`, sha256 `1e8246ed33bf36dc5c859351f7110f2cd31f98661989715c0fcf974ec48d3e2e` (M-02가 Google 저장소에서 받은 것과 같음). 같은 폴더의 lattice tflite 모델은 audio 모드에서 쓰이지 않음 |
| 표본율 | 48 kHz (기준·시험 모두; 원음 48 kHz, 디코드는 ffmpeg로 바로 48 kHz) |
| 채널 | 두 채널 평균 모노(합성 쌍은 처음부터 모노) |
| 스펙트로그램 (객체 속성) | `GammatoneFilterBank.num_bands = 32`, `min_freq = 50.0`, `speech_mode = False` |
| 패치 (객체 속성) | `ImagePatchCreator.patch_size = 30` — 기록된 패치 길이 0.6 s(= 30 × 20 ms) |
| 탐색 | `search_window = 60` |

## 쌍 (51)

- **T-97 공개 24 항목 × 2 디코드 = 48**. 비공개 항목 없음(D-0097).
  - `<item>.sori64`: T-97의 sori 64 명령(`t97-run.sh`의 `$DEF`, CBR 64). 인코더 main `7664225c`, mp3 바이트 md5 24/24가 T-97 `results/t97/pub/md5.txt`와 같다(`coded-md5.txt`).
  - `<item>.opus128`: ffmpeg 6.1.1 `-c:a libopus -b:a 128k`(VBR, T-97 `opus128`).
  - 정렬은 하네스 그대로: `r32_run.ref48`(원음 → 48 kHz) · `r32_run.aligned`(ffmpeg 디코드, 상호상관 지연 `lag`, 이득 맞춤 없음, 길이 맞춤), clip ±1, float WAV.
  - 확인: `moslqo`가 T-97 기록(`results/t97/raw`, 소수 넷째 자리)과 48/48 일치(|Δ| ≤ 4.6e-5 — 반올림 몫).
- **합성 3** (모노, 48 kHz, `t113_visqol_dump.py`가 만든다 — numpy 난수 대신 xorshift32라 Rust에서 그대로 재현 가능):
  - `syn-identical`: 기준 = 시험 = 220/440/1000/3000/8000 Hz 사인(진폭 .08/.05/.04/.02/.01) + 균등 잡음 0.01(시드 1), 5 s.
  - `syn-sine-noise`: 기준 = 1 kHz 사인 진폭 0.1; 시험 = 기준 + 사인 RMS보다 20 dB 낮은 백색 잡음(시드 2), 5 s.
  - `syn-pad-silence`: 기준 = `syn-identical`의 신호; 시험 = 무음 0.5 s + 기준 + 무음 0.5 s(6 s).
  - xorshift32: `s ^= s<<13; s ^= s>>17; s ^= s<<5`(32비트), 값 = `s / 2^31 − 1`.

## 파일

| 파일 | 내용 |
|---|---|
| `pairs.tsv` | 쌍마다 `ref_md5`·`deg_md5`, 프레임 수, 채널 수, `lag`, `moslqo`, `vnsim`, 패치 수 |
| `bands.tsv` | 쌍 × 32 대역: `center_freq`(`center_freq_bands`), `fvnsim`, `fvnsim10`, `fstdnsim`, `fvdegenergy` |
| `patches.tsv` | 쌍 × 패치: 기준·시험 패치 시작·끝 시각(s), `similarity` |
| `json/<pair>.json` | 위 전부 + 패치별 32 대역 `freq_band_means`·`freq_band_stddevs`·`freq_band_deg_energy`, 결과 객체 필드 목록 `result_fields` |
| `coded-md5.txt` | sori 64 mp3 바이트 md5 |

수치는 파이썬 `repr`(왕복 가능한 전체 정밀도).

- **md5는 WAV 파일 바이트가 아니라 표본 내용**(interleaved float32 little-endian, `data` 덩어리 내용)이다. libsndfile이 float WAV에 시각이 든 PEAK 덩어리를 써서 파일 바이트는 실행마다 다르다. 기준 md5는 같은 항목의 두 디코드에서 같다(24/24).
- 결과 객체(`SimilarityResult`)의 공개 필드: `center_freq_bands`, `fstdnsim`, `fvdegenergy`, `fvnsim`, `fvnsim10`, `moslqo`, `patch_sims`, `vnsim`. 패치(`PatchSimilarityResult`): `deg_patch_end_time`, `deg_patch_start_time`, `freq_band_deg_energy`, `freq_band_means`, `freq_band_stddevs`, `ref_patch_end_time`, `ref_patch_start_time`, `similarity`.

## 재현성

- 같은 입력으로 두 번 돌린 결과: 부동소수 117 039개 중 27개가 다르고 최대 |Δ| 2.7e-14(`moslqo`), 구조·패치 시각·md5는 같다. M-03 비교 허용치는 이보다 훨씬 커도 된다.
- 다시 돌리기: 코퍼스(`crates/conformance/corpus/fetch-odaq.sh` → `unzip` → `make-odaq-corpus.sh`, `make-extra-corpus.sh`), `cargo build --release -p conformance`, `target/qenv`(visqol-python soundfile numpy scipy), 그다음 `crates/conformance/scripts/t113-run.sh`(약 1.5분, 2코어).

## 출력에서 본 관계 (관찰 — 소스 아님, M-03 참고)

51쌍 모두에서 다음이 성립했다(차 0 또는 ≤ 1e-15):

- `vnsim` = 32 대역 `fvnsim`의 평균.
- `fvnsim[b]` = 패치들의 `freq_band_means[b]` 평균. `fvdegenergy[b]` = 패치들의 `freq_band_deg_energy[b]` 평균.
- `fvnsim10[b]` = 패치들의 `freq_band_means[b]` 중 **가장 낮은 ⌊패치 수/10⌋개(최소 1개)** 의 평균.
- 패치 `similarity` = 그 패치 32 대역 `freq_band_means`의 평균.
- `fstdnsim`은 위처럼 단순하지 않다: 패치 안 분산과 패치 사이 분산을 합친 √(평균(stddev² + (mean − 대역 평균)²))와 최대 0.007 차이 — 정확한 꼴은 미확인. 0.0인 대역이 잦다.
- 기준 패치는 0.28 s부터 0.6 s 간격으로 놓인다(`syn-*` 7개 = 5 s, 10 s 항목 16개). `syn-pad-silence`는 시험 앞에 0.5 s 무음이 있는데도 시험 패치 시각이 기준과 같게(0.28, 0.88, …) 기록된다 — 보고되는 시험 시각은 전역 정렬 뒤의 시간축으로 보인다. 그래도 `moslqo` 4.7317로 같은 신호(4.7321)와 거의 같다.
- 품질 범위: 공개 48쌍 `moslqo` 2.28(piano-vampire sori 64) – 4.73, `syn-sine-noise` 3.38, 같은 신호 4.732.

## 공개 리포 사본 (Public copy)

This copy leaves out the speech item and its values (24 → 23 public items, 51 → 49 pairs); the counts above describe the full development data set.
