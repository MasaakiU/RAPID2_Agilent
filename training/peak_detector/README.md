# ピーク検出モデル（前段）の学習

二段構成の前段（ピークトップの検出）を、合成データだけで学習するコード。仕様は [SPEC.md](SPEC.md)。後段（ピークの分離）の仕様の案は [SPEC_deconvolution.md](SPEC_deconvolution.md)。アプリ本体（`src/RAPID2_Agilent`）にも正解付けツール（`tools/peak_labeling`）にも依存しない。このフォルダだけで動く。

## ファイル

| ファイル | 中身 | 依存 |
|---|---|---|
| `simulator.py` | 合成データ（ピーク・ベースライン・ノイズ・アーティファクト）と正解 | numpy, scipy |
| `features.py` | 入力の 3 チャンネル（線形・log・欠測の印）。アプリでも同じものを使う | numpy |
| `targets.py` | 学習の正解（ヒートマップ・重み・ずれ・幅） | numpy |
| `data.py` | 学習データをその場で作る Dataset、固定の検証用データ | torch |
| `models.py` | `transformer`（A）、`unet`（A′）、`kensert`（K） | torch |
| `losses.py` | focal loss、回帰、ノイズ除去の補助 | torch |
| `decode.py` | 頂点の取り出し、正解との突き合わせ、AP などの評価 | numpy |
| `train.py` | 学習 | torch |
| `evaluate_real.py` | 正解付けツールのラベルでの、実データの評価 | torch |
| `export.py` | 重みを `.npz` に書き出す（アプリは torch なしで読む） | torch |
| `export_onnx.py` | 前段と後段を ONNX に書き出し、onnxruntime の出力が torch と同じか確かめる（アプリが torch なしで動かす） | torch, onnx, onnxscript, onnxruntime |
| `make_examples.py` | 合成データの例と図 | numpy, scipy, matplotlib |
| `deconvolve.py` | 後段（ピークの分離）の、モデル以外の部品：候補、形の高さの最小二乗、面積の不確かさの補正、評価 | numpy, scipy |
| `evaluate_deconvolution.py` | 後段の比べる基準（正解の形での最小二乗、EMG の当てはめ、アプリのやり方）と、学習した後段の、合成データでの評価 | numpy, scipy（窓の切り出しとモデルに torch） |
| `deconvolve_data.py` | 後段の学習データ（窓、誤りを足した候補、入力、候補ごとと窓の正解） | torch |
| `deconvolve_models.py` | 後段のモデル（候補ごとの形・本物らしさ・高さ、窓のベースライン・その他・ノイズ）と、窓への後段（モデルのあとに最小二乗） | torch |
| `deconvolve_losses.py` | 後段の損失（形、形の裾、本物らしさ、高さ、ベースライン、その他、ノイズ、最小二乗で出した面積） | torch |
| `train_deconvolution.py` | 後段の学習 | torch |
| `calibrate_deconvolution.py` | 学習した後段の、面積の不確かさ（σ）の補正を、合成データで作る | torch |
| `configs/` | 設定（前段：`transformer.json`、`unet.json`、`kensert.json`、動作確認用の `cpu_tiny.json`。後段：`deconvolution.json`、`deconvolution_unet.json`、`deconvolution_cpu_tiny.json`） | |

## 環境

Python 3.11 以上と、`requirements.txt` のパッケージ（torch は CUDA 版を入れる）。

```sh
pip install -r requirements.txt
python -m pytest -q          # 65 件、CPU で 30 秒ほど
```

## 学習

```sh
python train.py --config configs/transformer.json --out runs/transformer
python train.py --config configs/transformer.json --out runs/t_short --set steps=20000 workers=16
python train.py --resume runs/transformer/last.pt --out runs/transformer   # 続きから
```

- `--set key=value` で設定を上書きできる（値は JSON として読む。読めなければ文字列）。
- 出力（`--out`）：
  - `config.json`
  - `metrics.jsonl`：`log_every` ごとの損失と速さ、`eval_every` ごとの検証の結果（AP、しきい値ごとの適合率・再現率・F1、内訳ごとの再現率）
  - `last.pt`、`best.pt`（検証の AP が一番良かったもの）
- 検証用データは固定の乱数で作る（`validation` 件、既定 5000。1000 件では 1 回の AP が ±0.002 ほどぶれ、モデル同士の差と同じくらいになる）。学習のデータとは乱数の系列が別。

### データ作りの速さ

学習データは、学習しながら CPU のワーカーで作る。1 プロセスあたり毎秒約 180 件（2048 点、Apple M 系の CPU 1 コア）。GPU の 1 ステップ（バッチ 64）が数十 ms で終わるなら、`workers` を CPU のコア数近くまで上げないと、GPU が待つことになる。

`metrics.jsonl` の `steps_per_s` が、ワーカーの数を増やしても伸びない場合は、GPU 側が律速になっている。ワーカーを増やすと伸びる場合は、データ作りが律速になっている。

GPU マシン（RTX 5090、Threadripper PRO 7955WX 16 コア）で測った値：1 ワーカーあたり毎秒約 300 件。`transformer` は `workers=12` で GPU 律速の毎秒 49 ステップ（4 で 19、8 で 34）。2 台の GPU で同時に学習すると、CPU を分け合って、`workers=14` ずつでそれぞれ毎秒約 39 ステップ。`kensert` は GPU が速いので（GPU だけなら毎秒約 210 ステップ）、データ作りが律速になる（`workers=14` で毎秒約 60）。

numpy が MKL 版（Anaconda の numpy）だと、MKL が Intel の OpenMP を使い、torch の GNU OpenMP と同居して、ワーカーからのバッチの受け取りが 6 分の 1 の速さになる。`train.py` は、numpy を読む前に `MKL_THREADING_LAYER=GNU` を既定にしている（OpenBLAS や Accelerate の numpy には効かない・害もない）。

ワーカーの数が違うと、学習データの流れ（どのサンプルがどの順で来るか）も変わる。同じ `seed` と同じ `workers` なら、同じ流れになる。モデルを比べるときは `workers` もそろえる。

## 実データでの評価

`tools/peak_labeling` で正解を付けたデータセット（`xics.npz` と `xics.labels.json`）を使う。トレースを 0.5 秒ごとに補間し、確定済みで使えるトレースだけを評価する。データセットは複数渡せて、全体、種類ごと、データセットごとに集計する。

```sh
python evaluate_real.py --checkpoint runs/transformer/best.pt --dataset a.npz b.npz ... --out real.json \
    [--details found.json] [--step-seconds 0.5] [--tolerance-seconds 3] [--tolerance-fwhm 0.5]
python evaluate_real.py --app kensert --dataset a.npz b.npz ...   # アプリの検出器（kensert、peakonly、scipy）と比べる
```

- 正解は、採用した頂点と足した頂点。保留の頂点は採点しない（そこで検出しても、しなくても数えない）。
- 許容幅は、3 秒とピークの半値幅の半分の大きい方。
- `--app` はアプリの `RAPID2_Agilent.services.peak_detection` を、測ったままのトレースにかける（`src` が import できること）。Kensert は確率を確信度にし、0.05 以上を見る（アプリは 0.5 超を取る）。peakonly と scipy は確信度が無いので、すべて 1。
- `--details`：トレースごとに、正解の頂点（見つかった確信度、見つからなければ null）と、検出した頂点（true、false、保留に当たった ignored）を書き出す。

## 書き出し

```sh
python export.py runs/transformer/best.pt transformer.npz
```

## 合成データの例

```sh
python make_examples.py --count 100 --out examples
```

## 後段（ピークの分離）の比べる基準

仕様は [SPEC_deconvolution.md](SPEC_deconvolution.md)。合成データの同じ窓（`evaluate_deconvolution.SEED`）で、正解のピークごとの面積・形・範囲の誤差、不確かさの当たり具合、余計な候補が持った面積、見落としを、全体と内訳（隣との `Rs`、高さの比、S/N、形、サチュレーションなど）で出す。

```sh
python evaluate_deconvolution.py --method truth --count 2000 --out truth.json   # 正解の形での最小二乗（ノイズだけの誤差）
python evaluate_deconvolution.py --method emg --candidates perturbed --count 300 --out emg.json
python evaluate_deconvolution.py --method app --count 300 --out app.json
python evaluate_deconvolution.py --method emg --candidates front --front runs/2026-10-29_unet_v5_200k/best.pt \
    --confidence 0.3 --count 300 --out emg_unet_v5.json   # 前段の候補の、確信度 0.3 以上で
```

- `--candidates`：`oracle` は正解の頂点と幅、`perturbed` は前段の誤り（見落とし、ずれ、重複、何も無いところ）をまねたもの（`deconvolve.CandidateErrors`）、`front` は `--front` の前段（`train.py` の checkpoint）が実際に出したもの（しきい値 0.05 から）。`truth` は、いつも正解のピークごと。
- `--confidence`：候補のうち、この確信度以上のものだけを使う（アプリのやり方は、前段の頂点をしきい値で切って使う）。
- `emg` は、1 窓に 1〜4 秒ほどかかる。

## 後段の学習

前段の学習と同じく、合成データをその場で作りながら学習する。窓（`data.cut`）に、固定した前段（`front`、`train.py` の checkpoint）が実際に出した候補を付けて入れる。3割の窓（`rough`）では、その上に人工の誤り（`deconvolve.roughened`）を足す。前段とは候補の一覧だけでつながる。前段を学習し直したら、新しい前段の候補で評価し、悪くなっていれば続きを学習する（SPEC_deconvolution 4）。

```sh
python train_deconvolution.py --config configs/deconvolution.json --out runs/2026-10-30_deconv_v6_100k_unet_v5 \
    --set front=runs/2026-10-29_unet_v5_200k/best.pt workers=14
python train_deconvolution.py --resume runs/2026-10-30_deconv_v6_100k_unet_v5/last.pt --out runs/2026-10-30_deconv_v6_100k_unet_v5
python evaluate_deconvolution.py --method model --checkpoint runs/2026-10-30_deconv_v6_100k_unet_v5/best.pt \
    --candidates front --front runs/2026-10-29_unet_v5_200k/best.pt --count 300 --out model.json   # 基準と同じ 300 窓で
```

- `--set` は1つにまとめる（`--set front=... workers=14`。2つ書くと後のものだけになる）。
- `front` を付けないと、正解の頂点に誤りを足した人工の候補（`deconvolve.perturbed_candidates`）で学習する。前段の候補と違いすぎるので、試しとテストのためだけ。始めに出る行（`candidates of the front ...`）で確かめられる。
- `front` に一覧を渡すと（`--set 'front=["a.pt", "b.pt"]' workers=14`）、窓ごとにどちらか一方の前段の候補で学習する。検証の窓は、交互にそれぞれの前段の候補。

- 出力は前段と同じ（`config.json`、`metrics.jsonl`、`last.pt`、`best.pt`）。`metrics.jsonl` の `skipped` は、勾配が有限でなく飛ばしたステップの数（ずっと 0 のはず。増えていたら知らせてほしい）。
- 検証（`eval_every` ごと、既定 2000 窓）は、使うときと同じ道筋で行う。本物らしさが `real`（既定 0.5）以上の候補を取り、高さを `scipy` の非負の最小二乗で出す。`low`（既定 0.02）以上で `real` 未満の候補も最小二乗に入れ、その分は「その他」に足す（ピークとしては出さない）。
- 記録するもの：
  - 面積・形・範囲の誤差を、全体と内訳ごとに
  - 余計な候補が持った面積
  - 本物らしさの AP と、`real` での適合率・再現率
  - 各損失
- `best.pt` は、`objective` が一番小さいもの。`objective` は、ピークごとの `log(1 + |面積の相対誤差|)` の平均（見落としは誤差 1）と、余計な候補が持った面積の割合の和。
- 検証の窓の乱数（`deconvolve_data.VALIDATION_SEED`）は、比べる基準の窓（`evaluate_deconvolution.SEED`）とも、前段の検証とも別。
- 損失の重み（`loss_weights`）、引き寄せの強さ（`strength`）、本物らしさのしきい値（`real`）は、最初の当て推量。学習の結果を見て決める（SPEC_deconvolution 6）。
- `evaluate_deconvolution.py --method model` の `--real`、`--strength`、`--low` で、学習し直さずに、取る候補、引き寄せの強さ、その他として当てはめる候補を変えて比べられる（暫定は `--real 0.3 --strength 0.1`。`--low` を `--real` と同じにすると、その他としての当てはめは無し）。`low` より前の run の checkpoint は、`deconvolve_models.LOW` で評価する。
- ベースラインとその他は、入力の log チャンネルの目盛りで出す（`model_options` の `background`、既定 `"log"`）。それより前の run（`2026-10-30_deconv_v6_100k_*`）の checkpoint は、線形として読む。
- 速さ：エンコーダは前段の `transformer` と同じ。1 窓あたり、データ作りは前段（unet v5）を走らせるのと合わせて 34 ms ほど（Apple M 系の CPU 1 コア。前段だけで 20 ms）。バッチ 32 なら、`workers=14` で 1 秒に 13 バッチほど作れる。`steps_per_s` がそれに近ければ、データ作りが追いついていない。

## ONNX への書き出し

アプリは、前段と後段を onnxruntime で動かす（torch なし）。torch の書き出し（dynamo）で、CPU・float32 のまま書き出し、合成データの窓で、onnxruntime の出力・前段の候補・後段の面積が torch と同じかを確かめる。

```sh
python export_onnx.py --front runs/2026-10-29_unet_v5_200k/best.pt \
    --later runs/2026-11-02_deconv_v6_sure_100k_unet_v5/best.pt --out runs/onnx_unet_v5 --check 50
```

- 出力：`front.onnx`、`later.onnx`、`manifest.json`（元の checkpoint、入出力の名前、窓の長さ 2048、前段の stride、opset、確かめた結果）。最後に `as torch: yes` と出れば一致している（出力の差が 1e-2 未満、前段の候補がすべて同じ、面積の相対差の 99 パーセンタイルが 1% 未満）。
- 窓の長さは 2048 点に固定、後段の候補の数は 1〜1024 で可変。候補が 0 のときは後段を動かさない。
- 後段は、候補の mask を渡さず（`candidate_mask=None`）、窓 1 つの候補をすべて使う形で書き出す（値で形の変わる処理を通らない）。
- σ の補正（`calibrate_deconvolution.py` の JSON）は、ONNX とは別に、同じ組として置く。

## 後段の面積の不確かさの補正

最小二乗の σ は、ノイズの分しか表さず、実際の誤差の 1/20 ほどしかない（形や背景の誤りが入らない）。合成データで、ピークごとの当てはめの特徴（隣の形との重なり、隣の高さ、残差、S/N など、`deconvolve.fit_features`）と、実際の面積の誤差の関係を学び、σ を補正する。学習し直しはいらない。

```sh
python calibrate_deconvolution.py --checkpoint runs/x/best.pt --front runs/2026-10-29_unet_v5_200k/best.pt \
    --count 3000 --test 2000 --real 0.3 --strength 0.1 --low 0.02 --out runs/x/calibration.json
python evaluate_deconvolution.py --method model --checkpoint runs/x/best.pt --candidates front \
    --front runs/2026-10-29_unet_v5_200k/best.pt --real 0.3 --strength 0.1 --calibration runs/x/calibration.json
```

- 出すもの：`sigma`（誤差の 68% が入る幅）と `interval`（95% が入る幅）。誤差の分布の裾が重いので、2σ は 95% の幅にならない。最小二乗の σ は `sigma_noise` に残す。
- 補正は、後段・前段・`real`・`strength`・`low` の組ごとに作る（誤差がそれで決まるため。JSON に記録する）。
- 補正を学ぶ窓の乱数（`calibrate_deconvolution.CALIBRATION_SEED`）は、学習・検証・評価の窓と別。`--test` は評価と同じ窓（`evaluate_deconvolution.SEED`）で、内訳ごとの当たり具合を出す。
- 中身は、特徴の対数に線形な、誤差の絶対値の対数の分位点回帰（`deconvolve.fit_calibration`。scipy の線形計画）。重みは JSON に数十個の数で入る。

## コードの版と、学習結果の出どころ

- コードのフォルダ名に版や日付は入れない。コードは git の 1 か所だけにする。GPU マシンは同じリポジトリを clone して pull し、そこで直したコードはブランチにコミットして戻す。
- 学習のたびに、`config.json` の `provenance` に次を自動で記録する。重み（`.pt`）と書き出した `.npz` にも同じものが入る。
  - `git_commit`：このフォルダのコミット
  - `git_dirty`：未コミットの変更があったか（あったなら、そのコミットだけでは再現できない）
  - `simulator_version`：シミュレータの版（`simulator.py` の `VERSION`、SPEC.md の版と同じ）
  - `started`、`host`：日時とマシン
  - 続きから学習したときは、`resumed` にそのたびの記録が足される。
- シミュレータの出力（サンプルや正解の付け方）を変えたら、`VERSION` と SPEC.md の版を 1 つ上げる。版が違う学習結果同士は、検証用データも違うので、AP をそのまま比べない。
- 日付を入れるのは、出力のフォルダ名だけにする：`runs/2026-10-25_transformer_20k` のように、日付・モデル・目的を並べる。
