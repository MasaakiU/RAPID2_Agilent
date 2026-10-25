# Raw-scan retention-time alignment

Agilent `.d`を直接読み、chromatographic peak pickingを行わずに2 runを比較する
Obiwarp-like baselineです。追加したコード・テスト・実行例はこのフォルダ内にあります。
rawデータの読み込みと較正だけは、既存の`../io/agilent_reader.py`を共有します。
GUI、Qt、`AgilentDModel`には依存しません。

添付仕様の「1. データの特徴」に記載されたscan数、sample数、共通TOF軸、
共通polynomialなどは仮定していません。scanごとのm/zと強度をreaderから受け取り、
1 scanずつbinningします。既存reader内の検証付き較正キャッシュはそのまま利用します。

## 構成

| Module | 責務 |
| --- | --- |
| `io.py` | 既存readerへのadapter、MS level/polarity/RTの選択、単位変換 |
| `binning.py` | constant-ppm grid、sum/mean/max aggregation |
| `representation.py` | scan × m/zのfloat32行列構築 |
| `transforms.py` | none/sqrt/log1p/asinh、scale候補の提示 |
| `backend.py` | NumPy/PyTorchの差異、device選択、OOM時のchunk縮小 |
| `similarity.py` | cosine/Pearson、正規化、chunked matrix multiplication |
| `dtw.py` | 類似度行列だけを受け取るpath solverとdispatcher |
| `anchors.py` | ordered bijective anchorの抽出・間引き |
| `mapping.py` | query → referenceのPCHIP RT変換 |
| `config.py` | 設定。`OPTIONS`はCLIのフラグ・MCPの引数名から設定項目への唯一の対応表 |
| `pipeline.py` | 各stageをつなぐ実行入口。`map_from_similarity`は類似度行列以降の共通部分（GUIのB − Aマップも使う） |
| `runner.py` | CLIとMCPが共通に呼ぶ仕事（scale候補、アライメントの実行・保存・図） |
| `result_io.py` | 設定、RT変換、raw path、診断配列の保存・再読込 |
| `diagnostics.py` | 類似度、path、anchor、RT変換、補正量、XICの図 |
| `synthetic.py` | 既知のsmooth driftとrun固有の強いionを含むscan生成 |

## CPU環境での準備

repository rootから、プロジェクトを図の保存に使うmatplotlibとともに登録します。
Agilent native拡張もこのときビルドされるため、OSにC++ compilerが必要です。
GUI依存（`gui` extra）は入らないので、Qtが不要な実行環境でもそのまま使えます。
native拡張の詳細は`../io/README.md`を参照してください。

```sh
python -m pip install -e ".[diagnostics]"
```

CPU実行にPyTorchは不要です。

## 最初の実行

合成データは実ファイルなしで実行できます。`scale=1`はこの合成データの検証値です。
実データ向けに自動選択した値ではありません。

```sh
python -m RAPID2_Agilent.alignment synthetic --scale 1 --device cpu --output /tmp/synthetic_alignment
```

合成データの真値は出力先の`truth.json`に保存します。

実データのscale候補を確認します。以下は最初の32 MS1 scansのsubsetです。
median、percentile、低強度側のnoise proxyを表示しますが、どの値も自動採用しません。
noise proxyは機器のnoise floorの推定が保証された値ではありません。

```sh
python -m RAPID2_Agilent.alignment scales path/to/reference.d --max-scans 32
python -m RAPID2_Agilent.alignment scales path/to/query.d --max-scans 32
```

両runで共通のscaleを明示して実行します。例えば以下の`100`は説明用の指定値です。
実際にはデータ・aggregation・ppm幅を見て選びます。GPU環境へは元の`.d`フォルダを
渡せばよく、展開済みのraw配列をnpzへ変換・転送する必要はありません。

```sh
python -m RAPID2_Agilent.alignment align reference.d query.d --scale 100 --device cpu --output /tmp/run_alignment
```

その他の指定例:

```sh
python -m RAPID2_Agilent.alignment align reference.d query.d --scale 100 --ppm 20 --aggregation sum --metric pearson --max-scans 32 --scan-stride 5 --xic 150 151 --output /tmp/subset_alignment
```

`--scan-stride`はMS level/polarity/RTで選択した後のscan間引き、`--max-scans`はその後の
先頭scan数の制限です。これらは検証用で、全runを解析する場合は省略します。
詳細な設定は`examples/cpu.json`をコピーし、`--config settings.json`で渡せます。
明示したCLIオプションがJSONの設定を上書きします。`scale: null`のままasinhを実行すると
エラーになります。`--anchor-stride 5`などでanchorの密度も変更できます。

## Python API

```python
from RAPID2_Agilent.alignment import (
    AlignmentConfig, TransformConfig, align_agilent, save_result,
)

config = AlignmentConfig(transform=TransformConfig(method="asinh", scale=100))
result = align_agilent("reference.d", "query.d", config)
save_result(result, "alignment_result")
```

各処理は単独でも呼べます。

```python
from RAPID2_Agilent.alignment import (
    compute_similarity, solve_alignment, extract_bijective_anchors,
    fit_rt_mapping, apply_rt_mapping,
)

# XA/XB: aggregationとintensity transformを終えた行列
S = compute_similarity(XA, XB)
path_result = solve_alignment(S, method="obiwarp")
anchors = extract_bijective_anchors(path_result.path, S)
mapping = fit_rt_mapping(anchors, reference_rt_seconds, query_rt_seconds)
corrected_query_rt = apply_rt_mapping(mapping, query_rt_seconds)
```

`solve_alignment`はpath solverのdispatcherなので戻り値は`DTWResult`です。
RT補間は意図的に分離しています。`method="smooth_viterbi"`は拡張位置を確保し、
現在は`NotImplementedError`を返します。将来も同じ`S`をそのまま入力にできます。

`build_profile_matrix`の出力でraw binned matrixを、`transform_intensity`の出力で
変換後のmatrixを確認できます。pipelineでは大きな配列の複製を避けるため、
変換・正規化をin-placeで進めます。`keep_profiles=True`で残るのは最終正規化行列です。

## 数学的な選択

- 初期設定: 50–1700 m/z、10 ppm、**sum**、asinh、cosine、global DTW。
- binは`log1p(ppm * 1e-6)`間隔。区間は`[mz_min, mz_max)`で、最後のbinは狭くなる場合があります。
  10 ppmでは352,638 bins、1000 scansのfloat32行列は1 runあたり約1.41 GBです。
- `sum`はbin内の測定sample強度の総和を保持します。m/zでの積分値やsampling-density補正値では
  ありません。密度差による重み付けが最良かどうかは実データでの比較対象です。
- `mean`はzero-valued samplesも含めてsample数で割ります。`max`は最大sampleのみ。
  どの方式も空binは0です。aggregationを終えてから強度変換します。
- `sqrt`はsqrt(I)、`log1p`はlog(1+I)、`asinh`はasinh(I/scale)。scaleは明示指定が必要です。
- Pearsonはzero binsを含む全bin上でscanごとに中心化します。
  cosineのzero vector、Pearsonのconstant vectorの類似度は0とし、自己類似度を1にはしません。
  informative scanが全くないrunではpipelineが停止します。
- 類似度行列の行はreference、列はqueryです。保存・適用するRT変換は**query → reference**です。
- RTは内部・結果ファイル・band指定ともに**秒**。既存readerのRTは初期設定で分から秒へ変換します。
  local `.d/AcqData/MSTS.xml`の単位コメントとRT範囲を確認しています。
  異なるreaderの単位には`reader_rt_unit`を明示してください。
- MS1を初期選択とし、polarity混在時は明示選択を要求します。異なるpolarity/MS levelのrun比較は
  エラーにします。選択後のRTは厳密な昇順を要求し、元のscan IDを維持します。

DTWは`1-S`の最小コストを使い、対角重み2、水平/垂直重み1、各水平/垂直stepのgap penalty 0.1。
すべて変更可能です。累積コストとpredecessorを保持し、raw pathも保存します。
RT bandは初期設定では無効。`--rt-band-seconds 45`などで`|RT_A-RT_B| <= 45`秒の制限を付けられます。
このbandはmaskであり、計算量を減らす最適化はまだ行っていません。

anchorは、対角stepで区切られた水平/垂直transition blockごとに、局所類似度最大の候補を選びます。
同点ではpath上で早い方を採用します。defaultでは両端を保持し、その端点と同じ行/列の候補は
除外します。選択されたanchorは両軸で厳密に増加します。stride/max_anchorsで間引けます。

これはObiwarp-like baselineで、既存OBI-Warp/xcmsの厳密な移植ではありません。
global DTWは両端を対応付けるため、測定開始・終了範囲が異なるrunでは共通範囲を選ぶか、
今後のfree-end solverが必要になることがあります。PCHIP自体はslope/curvatureの罰則を最適化せず、
細かなanchorを残すと局所傾きに揺れが残ります。

補間範囲外はdefaultでNaNを返します。`extrapolation="error"`ならエラー、`"linear"`なら
両端の正のsecant slopeで線形外挿します。PCHIPの三次多項式をそのまま外挿することは避けています。

## GPU環境

`device="auto"`はPyTorchとCUDAが利用可能ならCUDA、それ以外はNumPy CPUを選びます。
`cpu`はNumPy、`torch_cpu`は検証用のPyTorch CPU、`cuda`はCUDAを要求し、利用不可なら
警告してNumPy CPUへfallbackします。実際に使用したbackend/deviceを結果に保存します。

PyTorchは[公式installation selector](https://pytorch.org/get-started/locally/)でGPU/driverに
対応するビルドを選んで追加してください。ここではCUDAのビルドを固定していません。
raw I/O・較正・binningはCPU、変換・正規化・類似度の行列積は選択したbackendで動きます。
CPU上に全profileを保持し、GPUへはrow chunkだけを送るため、profile全体がVRAMに入る必要はありません。
CUDA OOM時はchunkを半分にして再試行します。1 scanのchunkでも失敗する場合はエラーを返します。
multi-GPUやMPSには対応していません。CUDA kernel/driverの不整合をOOMとして隠すこともしません。

```sh
python -m RAPID2_Agilent.alignment align reference.d query.d --scale 100 --device auto --chunk-size 64 --output /tmp/gpu_alignment
```

CUDAの数値検証はこのマシンでは未実施です。`tests/test_similarity.py`にはCUDAがある場合だけ
実行されるCPU/GPU一致テストがあります。

## 保存結果と本命コードでの利用

```text
alignment_result/
    mapping.json       # RT変換のanchor、方向、単位、外挿方針、対象run情報
    metadata.json      # 設定、backend、ソフトウェアversion、stage別所要時間
    diagnostics.npz    # scan×scan類似度、raw path、累積cost、predecessor、anchor、RT、scan ID
    diagnostics.png    # CLIで生成される診断図
    xic_overlay.png    # --xicを指定した場合
```

ここでnpzに入るのは小さなscan-spaceの**結果**です。scan×m/z行列やraw spectrumは保存しません。
既存の結果ファイルは上書きしません。

本命コードからは`mapping.json`だけでも利用できます。対象runの照合用にパス・ファイル名・
metadata fingerprintを保存します。fingerprintはmetadataファイルを対象とし、巨大な
MSProfile.bin全体のcontent hashではありません。別runへ適用する前には対象を照合してください。

```python
from RAPID2_Agilent.alignment import load_mapping

mapping = load_mapping("alignment_result/mapping.json")
aligned_rt_minutes = mapping.transform(original_rt_minutes, unit="minutes")
```

元の`RT_list`を書き換えず、表示・scan選択側で利用するための結果です。
本命GUIへの画面追加は今回の変更範囲に含めていません。

## 検証

```sh
python -m pip install -e ".[diagnostics,dev]"
python -m pytest -q src/RAPID2_Agilent/alignment/tests
```

bin境界、sum/mean/max、transform順序、asinh scaleの明示、Pearson/cosineの数値、
NumPy/PyTorch CPUの一致、OOM再試行、DTWと全path列挙の最適解比較、bandの到達不能、
anchorの一対一性、PCHIPの方向・単位・単調性、保存結果のround trip、Qt/Torchなしの実行、
実Agilent scanの直接入力を検証します。

参考: [OBI-Warp原論文](https://pubmed.ncbi.nlm.nih.gov/16944896/)、
[SciPy PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)、
[PyTorch CUDA API](https://docs.pytorch.org/docs/stable/cuda.html)。
