# ピークの正解付けツール

ピーク検出モデルの評価用に、実データの XIC のピークトップへ正解を付けるためのツール。アプリ本体（`src/RAPID2_Agilent`）とは別物で、本体の関数を流用するが、本体からは使わない。パッケージにも入らない。

## 1. XIC を書き出す

```sh
python tools/peak_labeling/extract_xics.py RUN1.d RUN2.d ... --out xics.npz \
    [--targets targets.rpd.csv] [--spots 15] [--random 5] [--ppm 10] [--seed 0]
```

各ファイルの、測定全体の XIC を書き出す。

- `--targets`：`.rpd.csv`（または RAPID2 形式）のターゲット。アプリと同じ選び方（MS レベル、極性、m/z 窓）でトレースする。
- `--spots`：m/z-RT マップで強度の高い位置の m/z。±`--ppm` の窓で、MS1 のスキャンから作る。
- `--random`：ランダムな m/z。種類を広げ、ピークのないトレースを入れるため。

## 2. 正解を付ける

```sh
python tools/peak_labeling/label_peaks.py xics.npz [--labels xics.labels.json]
```

候補の頂点を出すので、良いものを選ぶ。

- 候補：Kensert らの CNN が出す頂点（確率 0.3 以上、数字はその確率）と、簡易版（scipy）の頂点のうち、ノイズの 5 倍以上目立つもの（「scipy」と表示）。
- 候補をクリック：採用（緑の▲）→ 却下（赤の×）→ 未決（灰色の○）と切り替わる。
- Shift + クリック：見落とした頂点を足す（青の▲）。近くの 3 スキャン以内でいちばん高い点に合わせる。足した頂点をクリックすると消える。
- A：確率 0.9 以上の未決の候補をまとめて採用。
- Done（Enter）：そのトレースを全体まで見終わったとして確定する。未決の候補は却下になり、次の未確定のトレースへ移る。
- N / P：次・前のトレースへ。
- Of no use：使えないトレース（壊れている、など）の印。Note は自由なメモ。

ラベルは変更のたびに保存される（既定ではデータセットの横の `xics.labels.json`）。候補はトレースを初めて開いたときに計算して保存するので、あとで検出の方法が変わっても変わらない。

## 決まりごと

- ターゲットかどうかに関係なく、トレースの中のピークはすべて付ける（blank の未知のピークも）。
- 「Done」で確定したトレースの、何も付いていない部分は「ピークなし」とみなす。全体を見終わってから確定する。
- 重なったピークは、頂点をそれぞれ付ける。

## ラベルの形式

```json
{"dataset": "...", "traces": {"0000": {
    "candidates": [{"rt": 3.29, "source": "kensert", "confidence": 1.0, "state": "taken"}],
    "added": [3.05], "done": true, "unusable": false, "note": ""}}}
```

`state` は `taken`（採用）、`left`（却下）、`undecided`（未決）。正解の頂点は、`taken` の候補と `added` を合わせたもの。

## テスト

```sh
QT_QPA_PLATFORM=offscreen python -m pytest tools/peak_labeling
```
