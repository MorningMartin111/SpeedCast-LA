# A 任务书｜METR-LA 公共数据、STGCN 与空间消融

**适用对象**　ECE1553 三人交通预测项目成员 A，以及协助 A 的 ChatGPT / Codex。
**你的交付**　让三人读取完全相同的数据，训练 STGCN，并测出显式邻居聚合给预测带来的变化。
**第一个可合并成果**　公共实验约定、一个可手算的小样本、能读取它的数据接口。这个 PR 一旦可用就发给 B、C。

## 直接开始

将本文件放进仓库或附在新对话里。以下两段任选适合当下工具的一段。

**给本地 Codex**

> 你协助我完成 ECE1553 项目的成员 A 工作。阅读本文件及当前仓库已有的 README、AGENTS 和 docs/protocol.md，确认仓库、当前分支和已有实现。按本文件的 A0–A4 推进。第一优先级是发布供 B、C 使用的公共协议、模拟数据和读取接口。随后完成真实数据、STGCN、消融、验证及交接。依据现有工程复用代码，按我的职责提交可运行的增量。每个阶段回报实际修改、运行过的命令、结果与下一个依赖。真实数据或 GPU 尚未就绪时，继续完成能在 CPU 和模拟数据上验证的工作。

**给 ChatGPT**

> 我是成员 A。请结合本文和我贴出的仓库状态，先定位我在 A0–A4 的哪一步，再解释该步目的、准备一段给 Codex 的执行指令，并告诉我应从它的输出核对哪些数字。协助我审阅数据泄漏、站点顺序和图卷积实现。讨论结果请区分实际测量、论文参考和待验证假设。

## A0｜建立三个人能同时出发的起点

1. 在团队选定的 GitHub 仓库建立工作分支。若负责人已建仓，直接使用其远端。确认本人 Git 身份和远端指向团队仓库。项目目标是从历史传感器车速预测未来速度，在 METR-LA 上对比 Persistence、STGCN、DCRNN、STAEformer，并展示拥堵分布。
2. 将本文件最后的“附录｜三人共用的实验约定 v1.0”放进 `docs/protocol.md`。建立轻量 Python 包 `traffic/` 和公共数据模块 `traffic/data/`。建仓的 README 暂列项目目标、共同入口和当前状态，最终报告由 C 整合。
3. A 维护 `traffic/data/`、`traffic/models/stgcn/`、`configs/stgcn*.json`、A 自己的测试和 `reports/models/stgcn.md`。B 维护 `traffic/eval/` 和 DCRNN，C 维护 STAEformer、`traffic/viz/` 和最终汇总。共享依赖由 A 先给基础版本，B、C 通过各自 PR 说明实际需要的新增项。记录各人实际跑通的环境。
4. 先交可以生成 120 行模拟数据的 `traffic.data.fixture`、读取接口 `load_bundle` 与最短使用示例。最先让 B、C 拿到一份可导入的数据包。真实数据源尚在下载时，这个 PR 也可以先合并。若 B、C 已按附录在开发，他们可直接用这份包替换自己的临时输入。
5. 验收起点数为 `61/1/13`。逐项核对 S02/S01/S03 顺序、08:00 对应 `o=96`、第 3 步对应 08:15、S01 目标缺测、S03 从 50 降到 20。把验证命令和 PR 或提交号发给队友。

## A1｜把真实数据做成唯一的团队输入

**输入资料。** 从 DCRNN 作者发布的 METR-LA 车速文件、配套传感器图与站点坐标开始。优先使用同一来源。记录实际文件名、获取地址、校验值、下载日期和上游版本。若使用 Kaggle 等镜像，用站点集合、时间轴、单位和路网内容验证与所选版本的对应关系。作者仓库给出了 HDF5 文件、构图方式和坐标入口。[DCRNN 作者数据说明](https://github.com/liyaguang/DCRNN#data-preparation)

将原始 HDF5 车速文件 `metr-la.h5`、洛杉矶传感器图 `adj_mx.pkl` 和洛杉矶站点坐标 `graph_sensor_locations.csv` 放在 `data/raw/`。如团队选定的镜像采用其他文件名，记录实际文件映射并在准备脚本中解析。按以下顺序实现 `python -m traffic.data.prepare`。

1. **读速度表。** 保存源表的时间语义和每一列的站点标识，输出原始维度、日期范围、重复时间数、间隔分布、每站有效率。通常看到 207 个站点和约 3.4 万个时间点。出现差异时明确原因，记录实际值。
2. **对齐图。** 从配套图读取有序站点列表、ID 到索引映射和邻接矩阵。显式解码可能出现的字节 ID，以字符串处理。选该图的 ID 顺序为规范顺序，用 ID 重排速度列、坐标和参考速度。检查 ID 唯一性、集合一致性、图尺寸、有限非负权重和坐标范围。用三个非相邻站点追溯原始速度列与图索引。坐标个别缺失可标记，保留其数值预测资格。
3. **切分与缺测。** 严格按附录用时间边界确定训练时段。仅在训练时段计算公共标量均值和标准差及每站点第 85 百分位参考速度。记录缺失率和前向填补比例。训练期没有任何有效观测或标准差为零等异常要报告。任何传感器不具备分级参考值时保留相应标记。
4. **建滑动窗口索引。** 按 `o` 生成 12 入、12 出的合法窗口。保存三组起点。逐组输出第一个和最后一个窗口的发出时刻、目标首末时刻以及样本数。训练和验证的未来标签时间集合应当互斥，验证与测试同理。对验证第一个窗口说明它可用哪些较早观测。
5. **写 bundle 和版本。** 按附录生成文件和公共 Dataset。指纹至少覆盖时间、节点顺序、图、速度及观测标记、填补、参考速度、划分、scaler 与字段语义。用规范的数组值、dtype 和有序文本计算，使复制目录或改变压缩容器时间戳不影响识别。原始速度单位和可视化单位统一为 mph。
6. **先发数据 PR。** 在 `docs/data.md` 给出准备命令、实际数据摘要、`data_id`、本地放置路径，以及下载或重建办法。团队成员可自行生成，也可取同一份数据包。此时 B、C 就可接入真实数据。模型代码另开 PR。

**A 自查用例**

- 将原速度表的列暂时打乱，按 ID 重新准备后的速度轴与原结果一致。用一个不对称的三节点图检查有向边方向。
- 模拟一个缺测标签和一个输入缺测。前者在 `y_mask` 中为假，后者使用之前有效读数；跳过时间缺口的窗口。验证起点的 `last_mph` 与原始最后有效读数一致。
- 在小样本副本更改测试段的未来值。训练数据、训练均值、训练参考速度及早于改动的历史输入保持逐项一致。确认这项检查覆盖预处理，而非只检查配置里的切分比例。
- 对同一个已知速度做标准化再还原，误差处于 float32 数值容差内。对真实数据打印首批输入和输出形状，检查全部日历下标范围。

## A2｜训练可解释的 STGCN 主模型

实现或适配 Yu 等人的时空图卷积模型。基本结构是时间门控卷积、跨节点图卷积、时间门控卷积组成的块，再接速度预测头。记录论文、具体实现来源、上游 revision、复用范围与团队改动。原作者实验的数据和输出设置应按论文描述引用，本项目的 12 步输出以实际实现为准。[STGCN 论文](https://www.ijcai.org/proceedings/2018/0505.pdf) · [作者代码](https://github.com/VeritasYin/STGCN_IJCAI-18)

- **图怎么用。** 公共有向图保留在 bundle 内。本模型可采用 `max(A,Aᵀ)` 对称化、去除对角线，然后构造对称归一化拉普拉斯。选择 Chebyshev 图卷积时记录阶数、缩放方法和零度节点的数值处理。同一图归一化操作只执行一次。
- **时间怎么用。** 每个输入只覆盖已观测的 12 步，模型沿时间轴卷积。选定时间核和层数后逐层检查时间长度仍为正。记录三个输入通道的映射、隐藏宽度、dropout、参数量和最终输出顺序。
- **预测头。** 推荐由每个站点的特征直接输出 12 个未来值，整形为 `[B,12,N]`。若适配的参考实现只支持单步，明确记录这次多步输出头改动。保持跨节点交互集中在空间层，在消融时能清楚说明拿掉了什么。
- **训练。** 用附录的公共训练和验证规则。可从 2 个 ST 块、时间核 3、隐藏宽度 32、Adam 学习率 0.001 开始，用验证集确定最终设置。种子和最佳 checkpoint 随运行保存。在有效标签上计算损失，使用与 B 一致的验证 `all12` MAE 选模型。
- **先做最小跑通。** 对模拟 batch 完成一次前向、反向与参数更新，然后用少量样本检查损失能下降。验证模式下保存、重载同一权重，对同一输入复算预测。接入真实数据后，先完成一个小批训练和导出，接着通过 B 的评测入口核对单位、样本数和轴顺序。

## A3｜空间消融只改变一个主要因素

实验问题是“给 STGCN 看到邻居信息，在当前数据和训练预算下带来多少收益”。复制主配置，命名为 `stgcn_temporal`。保留输入、时间卷积、输出头、隐藏宽度、数据顺序、种子和训练设置；把跨站点图卷积替换成对每个站点独立应用、输出宽度相同的线性投影或 1×1 通道变换，重新训练。

- 两个版本采用相同归一化设计，推荐按单站点特征通道归一化。审查输出头、归一化、全连接层，明确有无其他跨节点混合。
- 做一个局部依赖检查。在 eval 模式固定一个目标站点的输入，单独修改另一站点的历史。纯时间版该目标的预测应在数值容差内相同。完整版是否一定改变取决于边、参数和样本，以真实观测为准。
- 分别记录参数量和实际训练耗时。将完整版与消融版交给 B 的同一评测器，在 C 预定义的拥堵转变分组上复评。若预算只够一个种子，两版先共同使用 42，结果写为单次观察。

## A4｜入口、交接与个人提交

下面是要提供的仓库入口。每个命令都从仓库根目录、在项目环境中运行。选择消融时，配置与 run 目录同时切换。

```bash
python -m traffic.data.fixture --out data/fixture-v1
python -m traffic.data.prepare --raw-dir data/raw --out data/metr-la-v1
python -m traffic.models.stgcn.train --data data/metr-la-v1 --config configs/stgcn.json --seed 42 --run-dir runs/stgcn/seed42
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split val
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split test
python -m traffic.models.stgcn.train --data data/metr-la-v1 --config configs/stgcn_temporal.json --seed 42 --run-dir runs/stgcn_temporal/seed42
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn_temporal/seed42 --split test
python -m traffic.eval.run --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split test
```

给消融版也执行 val 导出与评分，并在配置锁定后评分 test。B 的导出模块尚未合并时可先验证内存形状与手算样本，模块就绪后使用公共写入函数生成最终文件。

| 交给谁 | 必交内容 | 对方可以怎样验收 |
|---|---|---|
| B、C，早期 | 公共约定、fixture 命令、数据加载接口、数据 PR 或 commit | 无需下载 METR-LA 即可读一个 batch，并找到 `o=96`。 |
| B、C，正式训练前 | 实际 `data_id`、站点顺序、切分摘要、数据准备命令或团队共享入口 | 本地报告同一指纹、同样的样本数和抽查值。 |
| C，最终 | STGCN 和消融的配置、checkpoint 地址及校验值、按种子组织的预测、验证/测试轻量指标、`reports/models/stgcn.md` | 使用 B 的入口复算指标，能说明消融改动和适用范围。 |
| 组员与 TA | A 本人分支和 PR、论文与实现引用、本人完成的部分、实际复现命令 | 能从仓库历史追溯数据、模型、消融和结论的来源。 |

没有仓库远端时可先完成本地实现与提交，远端就绪后由 A 本人推送。排查异常优先区分数据源、节点对齐、时间偏移、重复反标准化和模型实现。交接时写明已通过的检查、真实运行路径及尚待补充的证据。

## 在自己电脑上的第一次联调

公共 fixture 和相应模型入口可用后，先在仓库根目录尝试这一组短命令。计算预算仅用于验证完整路径，正式运行另建目录。

```bash
python -m traffic.data.fixture --out data/fixture-v1
python -m traffic.models.stgcn.train --data data/fixture-v1 --config configs/stgcn.json --seed 42 --run-dir runs/fixture/stgcn/seed42 --device cpu --max-epochs 1
python -m traffic.models.stgcn.predict --data data/fixture-v1 --run-dir runs/fixture/stgcn/seed42 --split val
python -m traffic.eval.run --data data/fixture-v1 --run-dir runs/fixture/stgcn/seed42 --split val
```

**本人的 PR 与交接消息建议。** 先用独立 PR 合并初始协议与 fixture，再用数据 PR 发布正式 bundle，最后从更新后的默认分支提交 STGCN 与消融。给组员的首条消息写明「共同起始 commit、fixture 命令、预期 61/1/13、数据接口位置」。第二条写明「数据 PR、真实 data_id、三组样本数、获取路径」。这样 B、C 可以在模型完成前启动长训练。

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
