## 附录｜三人共用的实验约定 v1.0

A 在建仓时把本附录原样放入 `docs/protocol.md`。B、C 可以先照本页做。若团队调整公共字段、时间含义或指标，更新该文件、版本和受影响的调用方，在 PR 里写明需重新生成的数据或结果。下列路径和命令是本项目约定实现的接口，开发完成后再在干净环境实际运行，并把验证过的命令写入 README。

### 1. 一次预测的含义

- 主实验为 METR-LA 的多站点车速回归。节点数由数据读取，正式数据预期为 207，小样本为 3。Persistence、STGCN、DCRNN、STAEformer 共用同一批预测起点。
- 令 `o` 为最后一个输入时刻在完整时间轴上的行号，`H=12`。输入是 `o-11 … o`，目标是 `o+1 … o+12`。08:00 发出预测时，第 1、3、6、12 步分别对应 08:05、08:15、08:30、09:00。Python 的 15/30/60 分钟下标为 `2, 5, 11`。
- 对按时间排序的 `T` 行，令 `a=floor(0.70*T)`、`b=floor(0.80*T)`，依次取 `[0,a)`、`[a,b)`、`[b,T)`。12 个目标必须全部落在该集合，输入可以使用之前已经发生的观测。仅保留这 24 个时间点均相隔 5 分钟的窗口。先划分时间，再在训练集合内部打乱样本。
- 按数据源保留时间语义。带时区的数据用真实经过的时间检查间隔，用当地时钟生成时段和星期。无时区的源索引保留其本地钟面时间，在 manifest 记录该假设。三人使用同一套时间字符串和整数行号。

### 2. A 发布的数据包与公共读取接口

正式数据目录约定为 `data/metr-la-v1/`，模拟数据目录为 `data/fixture-v1/`。二者文件结构相同。

| 文件 | 名称、类型与含义 |
|---|---|
| `manifest.json` | 至少包括 `protocol_version`、`data_id`、来源、时间语义、T、N、切分边界、每个集合的起点数、单位及缺测约定。`data_id` 由 A 的脚本根据有序时间与站点、逻辑数组和处理配置统一生成，排除本机路径、运行时间和压缩包时间戳。 |
| `timestamps.csv`、`sensors.csv` | 前者两列 `time_idx,timestamp_local`。后者三列 `sensor_id,latitude,longitude`，行顺序就是模型的节点轴。站点 ID 按字符串保存。 |
| `series.npz` | `speed_mph[T,N]`、`filled_mph[T,N]`、`ref_speed_mph[N]` 为 float32，`observed[T,N]` 为 bool。`tod_slot[T]` 表示 0–287 的五分钟时段，`dow[T]` 表示周一 0 至周日 6。 |
| `graph.npz`、`splits.npz` | `adj_dir[N,N]` 为 float32，`adj_dir[i,j]` 表示原始路网中从 i 到 j 的权重。`train`、`val`、`test` 均为按升序保存的 int64 预测起点。 |
| `scaler.json` | `mean_mph`、`std_mph` 来自训练时段全部有效车速，使用总体标准差。输入标准化、反标准化和跨模型比较使用同一组数。 |

公共处理约定如下。

1. 在选定的 METR-LA 版本中，把有限且大于零的原始车速记为有效观测。`speed_mph` 缺测处存 0 占位，`observed` 单独记录真假。训练、验证、测试和作图均保留原始标签有效性。
2. 各站点按时间向前填充输入。每个连续时间段开始时若尚无有效历史，以训练均值兜底。验证和测试可承接前一时段已经发生的有效观测。`filled_mph` 仅用于输入和 Persistence。每站点的 `ref_speed_mph` 为该站训练期有效速度第 85 百分位，采用线性分位数，无有效观测则存 NaN 并在分级时排除。
3. `from traffic.data.dataset import load_bundle`。`bundle = load_bundle(path)` 暴露 `sensor_ids, timestamps, adj_dir, ref_speed_mph, origins, mean_mph, std_mph, data_id`，其中 `origins` 按 split 名索引。`bundle.dataset("val")` 返回按起点顺序读取的 PyTorch Dataset，另两个可用分组名为 `train`、`test`。
4. 单样本字典包含 `x[12,N,3]`、`y_mph[12,N]`、`y_mask[12,N]`、`last_mph[N]`、`origin_observed[N]` 和标量 `origin_idx`。DataLoader 在最前面加 batch 轴。`x` 的三个 float32 通道依次为标准化后的历史车速、`tod_slot/288`、以浮点存放的整数 `dow`。`last_mph` 取起点处的 `filled_mph`。`origin_observed` 取起点处原始观测标记。预测函数只接收其在预测时已经能看到的信息。

### 3. B 发布的预测文件与评分接口

- 各模型的运行目录为 `runs/<model>/seed42/`，模型名用 `stgcn`、`dcrnn`、`staeformer`。消融另用 `stgcn_temporal`，Persistence 用 `runs/persistence/fixed/`。每个训练所得模型保存 `config.json`、`run.json`、`best.pt`、逐轮训练/验证记录。根据执行阶段导出 `val.npz`、`test.npz`。
- 每个预测包使用 `pred_mph[S,12,N]` float32、`origin_idx[S]` int64、`origin_time[S]` 字符串、`sensor_id[N]` 字符串、标量字符串 `split`。`S` 为该集合全部起点。数组按数据包的起点与站点顺序导出。字符串用 NumPy Unicode 类型，可用 `allow_pickle=False` 读取预测包。每个预测值均为有限的原始 mph。
- `run.json` 记录 `model, seed, protocol_version, data_id, units, code_commit`，以及配置、实现来源、环境、最佳 epoch、验证 MAE、权重位置或获取方式。Persistence 的 seed 和训练特有字段取 null。必要时补上权重及预测文件的 SHA-256。若运行基于尚未提交的改动，先记录事实，再用已提交的同一实现确认要交付的结果。
- 共用导出函数为 `traffic.eval.io.write_predictions(run_dir, split, bundle, origin_idx, pred_mph)`，评分函数为 `traffic.eval.metrics.masked_scores(y_true_mph, y_pred_mph, mask)`。模型负责人保证自己的预测轴确实对应传入的 ID。B 的评测器从 A 的数据包独立读取标签和掩码，并校验数据版本、单位、形状、ID 与起点的逐项顺序。
- 对有效元素定义 `e = 预测mph − 实测mph`。`n` 是有效标量个数，`MAE = mean(abs(e))`，`RMSE = sqrt(mean(e²))`，`bias = mean(e)`。正偏差表示高估车速。先用布尔掩码选中元素，再按总数汇总；全缺测分组报告 `n=0` 和空指标。训练遇到全缺测 batch 时跳过该次损失更新并计数。
- `metrics_val.csv` 或 `metrics_test.csv` 至少包含 `model,seed,split,horizon_step,n,mae_mph,rmse_mph,bias_mph`。`horizon_step` 取 `3,6,12,all12`。前三项按**该步**在所有起点与站点上评分；`all12` 汇总全部 12 步有效元素。计算总体 RMSE 时先合计平方误差，最后开根号。

### 4. 实验与交付约定

- 三个主模型从随机权重开始。共同训练目标为有效标签的 MAE，模型内部可以输出标准化速度，但损失、监控、导出须明确单位，指标使用 mph。监控每个 epoch 的验证 `all12` MAE 并恢复最佳权重。默认首轮上限 60 epochs、早停耐心 10 次验证、首个种子 42。学习率、隐藏维度和适合显存的 batch size 可按模型选择，记录实际配置、调参次数与耗时。需要改变预算时在结果表注明。
- 先完成共同种子 42 的完整实验。资源允许时三人再共同补 43 和 44。报告每个种子的结果，再对同一组种子报告均值和样本标准差。一个种子只报单次结果。Persistence 是确定性的，运行一次即可。所有调参、checkpoint 选择和排错优先使用模拟或验证集，模型配置确定后再进行正式测试。
- 原始数据、处理后数组、完整预测和权重可放团队共享位置。Git 中至少有复现脚本、实际配置、轻量指标、图、结果索引和每人的贡献记录。结果索引提供团队可访问的位置、文件名与校验值。正式结果来自实际执行的日志和输出。三名成员各自使用自己的 GitHub 账号提交本人负责的工作、推送分支、建立 PR。

### 5. 三人都能手算的小样本

以下定义足以在 A 的数据包发布前开展模型和评测开发。模拟值仅用于接口验证。

- `T=120`、`N=3`，本地时间起于周一 `2012-03-05 00:00`，每行 5 分钟。站点轴依次为 `S02,S01,S03`。有向边为 `S02→S01` 权重 1、`S01→S03` 权重 0.5，其余为 0。模拟经纬度依次为 `(34.00,-118.30)`、`(34.01,-118.29)`、`(34.02,-118.28)`。
- 行号 `t=0…119`。S02 速度为 `40+t/4`。S01 平时为 60，在 `t=96,99` 缺测。S03 平时为 50，在 `t=89…91` 以及 `t=99…108` 为 20，两个区间均含端点。切分边界为 `84,96`；合法起点分别为 `11…71`、`83`、`95…107`，共 `61/1/13` 个。
- 单独检查起点 `o=96`（08:00）的第 3 步（08:15）。Persistence 预测为 `[64,60,50]`，真实值为 `[64.75,缺测,20]`。期望 `n=2`、`MAE=15.375`、`RMSE=sqrt(450.28125)`、`bias=14.625`。这些数字只对应这一行、这一步。S03 的参考速度为 50，实测从 50 降到 20，可用于验证拥堵分组与地图着色。

### 6. 从各自电脑接入同一仓库

A 在仓库登记一个已跑通的 Python 版本、`requirements.txt` 和项目根目录的运行方式。首次开发可选 Python 3.11 系列；以团队实际验证的依赖组合为准。每人在自己的克隆中创建虚拟环境，可用 `python -m venv .venv`。Windows PowerShell 可用 `.\.venv\Scripts\Activate.ps1` 激活，WSL/Linux 可用 `source .venv/bin/activate`，再执行 `python -m pip install -r requirements.txt`。需要 GPU 的训练机选用与本机兼容的 PyTorch 构建，记录 `torch.__version__`、`torch.version.cuda` 和 `torch.cuda.is_available()` 的实际输出。模拟开发应支持 CPU。

训练命令支持 `--device cpu` 与 `--max-epochs 1`，供本地小样本联调。实际生效的参数一起写入该次 `config.json`。实现初期只跑自己的模块和必要公共接口。公共数据 PR 与公共评测 PR 优先合并到默认分支，三人各自在工作分支同步。正式训练可以各用自己的机器，合写报告时按同一数据和评测程序复算。

### 7. 场景规则及个人 Git 交付

对每站以有效车速除以其训练期参考速度，比例 `r>=0.80` 为畅通、`0.50<=r<0.80` 为缓行、`r<0.50` 为拥堵。`free_to_congested` 与 `free_to_free` 由起点和目标的实际状态定义。只在两端真实观测有效且参考速度有效的样本中作此比较，全体有效目标继续进入主表。C 实现场景分组，B 的评分函数处理各组误差。正式配置和阈值在查看测试结果前固定。

三人共用一个团队仓库，使用各自的 Git 身份。例如从共同默认分支建 `feat/a-data`、`feat/b-eval` 或 `feat/c-staeformer`，后续模型 PR 可以从合并公共接口后的默认分支创建。每位成员在自己的工作树查看 `git status -sb`，仅暂存本次负责的文件，查看 `git diff --cached --stat`，然后提交并用 `git push -u origin HEAD` 推送本人分支、创建 PR。仓库用 `.gitignore` 管理本地环境、原始大数据、运行权重和完整预测，保留轻量指标、图与获取索引。交接附上本人 PR/commit、实际运行命令、数据指纹、结果路径和已知限制。
