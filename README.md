# RAPID2_Agilent

Agilent LC-MS の `.d` フォルダをベンダーソフトなしで直接読む、ビューアと解析ツールです。

## インストール

repository root で、用途に合わせて extra を選びます。

```sh
python -m pip install -e ".[gui,diagnostics,dev]"    # GUI、アライメントの図、pytest
```

| extra | 内容 |
| --- | --- |
| `gui` | PyQt6, pyqtgraph |
| `diagnostics` | アライメント結果の図（matplotlib） |
| `gpu` | アライメントの CUDA 計算（PyTorch） |
| `dev` | pytest |

native 拡張（`io/_native` の Cython/C++）はインストール時にビルドされるので、C++ compiler が必要です。
ソースを変更したら `python setup.py build_ext --inplace --force` で再ビルドします。

## 起動

```sh
python run.py [run.d ...]            # インストール後は rapid2-agilent [run.d ...] でも可
python run.py --centroid run.d ...   # centroid スペクトルだけを読んで開く（サーバー上の run 向け）
python run_app_with_test_data.py     # tests/data のテストデータを開いた状態で起動
```

解析だけなら GUI なしでコマンドラインからも使えます。結果は JSON で出力されます。

```sh
rapid2-agilent-cli --help
rapid2-agilent-cli extract-xic --path run.d --mz 150.5 --tolerance 0.5 --unit m/z
rapid2-agilent-cli --format tsv -o xic.tsv extract-xic --path run.d --formula C5H13NO --adduct "[M+H]+" --max-points all
rapid2-agilent-cli quantify-targets --runs a.d b.d --targets-file targets.rpd.csv --rpd-csv areas.rpd.csv
rapid2-agilent-cli extract-xic --path /Volumes/server/run.d --mz 150.5 --spectra centroid
python -m RAPID2_Agilent.alignment --help    # アライメントの全オプションと合成データでの検証
```

`--format tsv` は答えの表（trace、ピーク、行など）だけを書き出し、`-o` はファイルに書きます。
null を取るオプションは `all` / `none` / `null` で null になります（`--max-points all` は間引かない）。
オブジェクトの引数（`--targets`）は JSON で書きます。
`--spectra centroid` は、プロファイル（MSProfile.bin）ではなく、取り込み時に保存された centroid
スペクトル（MSPeak.bin、約 1/17 の量）だけを読みます。サーバー上の run を速く読めますが、強度はピークの
高さなので、XIC や面積はプロファイルの 0.15〜0.4 倍ほどになります。比べるのは同じ spectra で読んだ値どうし
にしてください。アライメントはプロファイルだけを読みます。
`--path` や `--runs` には RAPID2 の `.rpd` ファイルも渡せます（MS1 のスキャンだけ。アライメントは `.d` だけ）。
`.rpd` は profile か centroid のどちらか一方を持ち、`--spectra` を省くとそれで読みます。

## テスト

```sh
python -m pytest
```

`tests/data` の生データは git に含めていません。データがない環境では、それを使うテストは skip されます。

## 構成

| パス | 役割 | Qt |
| --- | --- | --- |
| `io/` | `.d`・`.rpd` の reader と native 拡張。`io.MSData` は reader と GUI の model が共通に持つメソッド | 不要 |
| `alignment/` | peak picking なしの RT アライメント（[README](src/RAPID2_Agilent/alignment/README.md)） | 不要 |
| `models/chemistry.py`, `target.py`, `mz_rt_map.py`, `mz_rt_diff.py` | 化学式、ターゲット、m/z × RT マップと差分 | 不要 |
| `models/*_model.py` | GUI 用の Qt model | 必要 |
| `services/operations.py` | 1 run への操作（スキャンの選択、XIC・TIC・BPC、平均スペクトル）。numpy 配列を返す | 不要 |
| `services/quantification.py`, `rpd_csv.py` | ターゲットの面積などの計算と、`.rpd.csv` の書き出し・読み込み（[仕様](docs/rpd_csv_format.md)） | 不要 |
| `services/maps.py` | m/z × RT マップの強いスポットと、2 run の差（B − A）の大きいスポット | 不要 |
| `services/reports.py` | 同じ操作の JSON 向けの要約（CLI・MCP 用） | 不要 |
| `mcp/` | AI アシスタント（MCP）の窓口。MyWB と同じ構成 | `bridge`、`controller` のみ |
| `cli.py` | `rapid2-agilent-cli`。MCP の解析ツールから自動生成 | 不要 |
| `main.py`, `presenters/`, `views/` | GUI | 必要 |

### 処理を 1 か所に書くために

GUI・CLI・MCP は入口が違うだけで、処理は同じものを呼びます。新しい処理は次の場所に書きます。

| 書くもの | 場所 | 呼ぶ側 |
| --- | --- | --- |
| 1 run への操作（何を抽出するか） | `services/operations.py` | GUI の presenter、`reports` |
| CLI・MCP 向けの要約（間引き、上位ピーク） | `services/reports.py` | MCP の解析ツール |
| CLI・MCP の解析コマンド | `mcp/tools.py` の `analysis_tools()` に 1 つ登録 | MCP と `rapid2-agilent-cli` の両方に自動で出る |
| 画面の操作（ファイルを開く、表示を変える） | presenter のメソッド | GUI のメニュー、MCP のビューアツール |
| アライメントの実行と保存、オプションの対応表 | `alignment/runner.py`、`alignment/config.py` の `OPTIONS` | 両方の CLI と MCP |

presenter に書いてよいのは、ウィジェットの状態を操作の引数にすることと、結果を描くことだけです。
意図的に 2 か所に分けている箇所には、その場所に `NOTE:` で理由を書いています（`models/mz_rt_diff.align_maps`、
`models/rpd_model.BaseMSDataModel`、MyWB からコピーした `mcp/`）。

## AI アシスタント（MCP）

MyWB と同じ構成です。起動中の RAPID2 ウィンドウが MCP サーバーになり、AI クライアントが起動する小さな
relay が、その窓口まで stdio のメッセージを中継します。ウィンドウはループバックの HTTP だけで待ち受け、
接続には起動ごとに作るトークンが必要です。

### 使い方

1. RAPID2 で **Tools > AI Assistant Access** をオンにします。設定は記憶され、次回からは起動時にオンになります。
   ウィンドウのタイトルに、そのウィンドウの文字（`RAPID2 Agilent — A` など）が出ます。
2. **Tools > Copy AI Assistant Setup Command** から使うクライアント（**Claude Code** か **Codex**）を選び、
   コピーされる 1 行をターミナルで一度だけ実行します。

   ```sh
   claude mcp add rapid2-agilent -- ~/Library/Application\ Support/RAPID2_Agilent/bin/rapid2-mcp-relay   # Claude Code
   codex mcp add rapid2-agilent -- ~/Library/Application\ Support/RAPID2_Agilent/bin/rapid2-mcp-relay    # Codex
   ```

   この launcher は RAPID2 が起動のたびに書き直すので、Python 環境を変えても登録し直す必要はありません。
   Codex の登録は `~/.codex/config.toml` の `[mcp_servers.rapid2-agilent]` に書かれ、Codex の CLI・IDE 拡張・
   ChatGPT アプリの Codex が同じ設定を読みます。
   Claude Desktop では同じパスを `claude_desktop_config.json` の `mcpServers` に `command` として書きます。
   環境の実行ファイル `rapid2-agilent-mcp` を直接登録しても動きます。

コマンドの代わりに、次のプラグインからも登録できます（どちらか一方で十分です）。どちらも RAPID2 本体を含まず、
上の launcher を起動するだけなので、RAPID2 を更新しても入れ直す必要はありません。先に 1. を一度済ませてください。

- **Claude Desktop**: [`integrations/claude-desktop/rapid2-agilent.mcpb`](integrations/claude-desktop/rapid2-agilent.mcpb)
  を開くか、Settings > Extensions からインストールします。
- **Codex**（CLI・IDE 拡張・ChatGPT アプリ）: このリポジトリが marketplace です。

  ```sh
  codex plugin marketplace add /path/to/RAPID2_Agilent   # clone した場所（公開後は MasaakiU/RAPID2_Agilent）
  codex plugin add rapid2-agilent-macos@rapid2-agilent    # Windows では rapid2-agilent-windows
  ```

プラグインは `scripts/` のスクリプトが生成し、生成物をコミットします。`pyproject.toml` の `version` を上げたら
作り直します（古いままだと `tests/test_codex_plugin.py` と `tests/test_claude_desktop_mcpb.py` が落ちます）。

```sh
python scripts/build_codex_plugin.py          # .agents/plugins/marketplace.json と integrations/codex/
python scripts/build_claude_desktop_mcpb.py   # integrations/claude-desktop/rapid2-agilent.mcpb（npx があれば mcpb validate も）
```

Windows では launcher が `.cmd` なので、どちらも `cmd /c` 経由で起動します。Windows 上での動作はまだ確かめていません。

RAPID2 を閉じていても、一度でもオンにしたことがあれば、relay がツール一覧に答え、解析ツールは relay が
自分で実行します（ウィンドウが開いていればウィンドウが答えます）。ビューアのツールにはウィンドウが要るので、
`rapid2_launch` で RAPID2 を起動できます。起動しただけでは AI アクセスはオンになりません。

### ツール

| 種類 | tool | 内容 |
| --- | --- | --- |
| ビューア | `rapid2_get_status` | 開いているファイル、選択中の RT・m/z 範囲、trace type、MS level |
| | `rapid2_open_files` | `.d` か `.rpd` を開く（`.d` は `spectra: centroid` で centroid スペクトルだけを読む） |
| | `rapid2_set_view` | RT・m/z 範囲、trace type、MS level を表示する（View > Undo View Range で戻せる） |
| | `rapid2_capture` | ウィンドウかプロットの PNG |
| | `rapid2_list_targets` / `rapid2_add_target` / `rapid2_select_target` | ターゲットの一覧、追加、選択して表示（Edit > Undo Target Edit で戻せる） |
| | `rapid2_import_targets` / `rapid2_export_targets` | Load Targets / Export Targets と同じ読み込み・書き出し |
| 解析 | `rapid2_find_runs` / `rapid2_describe_run` | `.d` と `.rpd` の列挙（`.d` はサンプル名・Sample Type・位置・取り込み時刻つき、`order: acquired` で測定順）、`sample_info.xml` の全項目・scan 数・RT 範囲・scan type、持っている spectra とその大きさ |
| | `rapid2_extract_xic` / `rapid2_extract_trace` | XIC（m/z か formula + adduct）、TIC・BPC（apex と、極大を残して間引いた trace） |
| | `rapid2_extract_spectrum` | RT 範囲の平均・合計スペクトルの強いピーク |
| | `rapid2_calculate_ion_mz` | 化学式と adduct から m/z |
| | `rapid2_calculate_isotope_pattern` | 化学式と adduct のイオンの同位体パターン（M+k ごとの平均 m/z と存在比、天然存在比は NIST、`[13C]`・`[2H]` で書いた原子はその同位体だけ） |
| | `rapid2_quantify_targets` | ターゲット（`.rpd.csv` か引数）の各 run での面積・高さ・ピークトップ。`.rpd.csv` にも書ける。各行に見直すべき点の警告（ピークトップが RT 窓の最初・最後のスキャン、`saturation_height` を渡せばそれ以上の高さ）、警告のある行の一覧を `flagged` に返す |
| | `rapid2_map_spots` / `rapid2_compare_runs` | m/z-RT マップの強いスポット、2 run の差（B − A）の大きいスポット |
| | `rapid2_suggest_alignment_scales` / `rapid2_align_runs` | 2 run の RT アライメント（結果はファイルに保存） |
| その他 | `rapid2_list_windows` / `rapid2_launch` | 開いているウィンドウ、RAPID2 の起動 |

ビューアのツールは Qt のメインスレッドで実行され、ウィンドウが複数あるときは `instance`（タイトルの文字）で
対象を指定します。解析ツールはファイルのパスを受け取り、ウィンドウには触れず、リクエスト側のスレッドで
計算するので、長いアライメントでも GUI は固まりません。`rapid2_export_targets` も、ターゲットと開いている
ファイルをメインスレッドで受け取ったあと、計算はリクエスト側のスレッドで行います。

アライメント以外の解析ツールは、`.d` フォルダと `.rpd` ファイルのどちらのパスも取り、`spectra`（`profile` か
`centroid`）を取ります。省くと profile を読み、centroid しか持たない run（`.rpd` はどちらか一方だけを持つ）は
その centroid を読みます。複数の run を読む `rapid2_quantify_targets` と `rapid2_compare_runs` では、最初の
run（reference）で決まった spectra で全部を読みます。アライメントは `.d` のプロファイルだけを読みます。
`rapid2_get_status` は開いている各ファイルの `spectra` を返し、`rapid2_export_targets` は各ファイルを
ウィンドウと同じ spectra で読みます。

XIC・TIC/BPC・スペクトルは GUI と同じスキャンを使います。adduct（または `polarity`）の極性のスキャンだけを使い、
両方の極性を含む run では極性の指定が必要です。XIC で m/z 窓にデータ点がないスキャンは、0 ではなく欠測として
除き、その数（`n_missing`）とプロットと同じ警告を返します。
