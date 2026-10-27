# ピーク検出モデル（前段）の学習

二段構成の前段（ピークトップの検出）を、合成データだけで学習するコード。仕様は [SPEC.md](SPEC.md)。アプリ本体（`src/RAPID2_Agilent`）にも正解付けツール（`tools/peak_labeling`）にも依存しない。このフォルダだけで動く。

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
| `make_examples.py` | 合成データの例と図 | numpy, scipy, matplotlib |
| `configs/` | 設定（`transformer.json`、`unet.json`、`kensert.json`、動作確認用の `cpu_tiny.json`） | |

## 環境

Python 3.11 以上と、`requirements.txt` のパッケージ（torch は CUDA 版を入れる）。

```sh
pip install -r requirements.txt
python -m pytest -q          # 27 件、CPU で数秒
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
