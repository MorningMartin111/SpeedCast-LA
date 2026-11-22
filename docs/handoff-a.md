# A 的交接与个人贡献

公开仓库：[MorningMartin111/SpeedCast-LA](https://github.com/MorningMartin111/SpeedCast-LA)。公共数据接口、正式数据准备和 STGCN/空间消融三个 PR 均已合并，团队 `main` 已具备共同运行起点；本次同步的 `main` 提交为 `9900a3d06b5327b527b69082f255144201603b9d`。

本次正式训练报告和轻量证据位于独立结果分支 `feat/a-training-results`，结果 [PR #4](https://github.com/MorningMartin111/SpeedCast-LA/pull/4) 已提交，待审阅；该结果 PR 合并前，`main` 的模型报告仍是此前的 CPU 联调版本。下方训练结论对应本结果分支，代码与数据准备入口已经在 `main` 可用。

| 顺序 | PR | 当前状态 / 内容 |
| --- | --- | --- |
| 1 | [公共协议、fixture、读取接口](https://github.com/MorningMartin111/SpeedCast-LA/pull/1) | 已合并；共同接口最初提交 `7b281b4efb68de5bdc8a555625dec783dd989ef1` |
| 2 | [正式 METR-LA 数据](https://github.com/MorningMartin111/SpeedCast-LA/pull/2) | 已合并；来源校验、准备脚本、正式审计 |
| 3 | [STGCN 与空间消融](https://github.com/MorningMartin111/SpeedCast-LA/pull/3) | 已合并；模型、训练/推理、验收与历史联调报告 |
| 4 | [种子 42 正式训练结果](https://github.com/MorningMartin111/SpeedCast-LA/pull/4) | 已提交，待审阅；文档与轻量训练证据 |

B 可从 `main` 开始 DCRNN 和统一评测，C 可开始 STAEformer、地图与拥堵分组；已有开发分支先同步 `origin/main`。无需等待最终预测包即可读取同一数据包。

## 可直接发给 B、C 的交接消息

> B、C：公共接口、正式数据与 STGCN/空间消融已分别在 PR [#1](https://github.com/MorningMartin111/SpeedCast-LA/pull/1)、[#2](https://github.com/MorningMartin111/SpeedCast-LA/pull/2)、[#3](https://github.com/MorningMartin111/SpeedCast-LA/pull/3) 合并到 `main`（`9900a3d`）。读取入口为 `traffic.data.dataset.load_bundle`，最短 batch 示例见 README。fixture 预期起点数 `61/1/13`、站点顺序 `S02,S01,S03`；`o=96` 为 08:00，第三步 08:15，手算 MAE 为 15.375 mph。正式 METR-LA 指纹 `sha256:61a74eeffbe6f6c22d44ed1dbf02f3032ccd14ea88c09c61aa753f427e494219`，`T=34272,N=207`、起点数 `23967/3416/6844`。请核对相同指纹后开始各自模型；B 可接 DCRNN/统一评分，C 可接 STAEformer/地图/拥堵分组。两版 STGCN 的种子 42 完整训练已完成，结果 [PR #4](https://github.com/MorningMartin111/SpeedCast-LA/pull/4) 已提交，待审阅，尚未合并到 `main`；验证记录见独立结果分支的 [STGCN 报告](https://github.com/MorningMartin111/SpeedCast-LA/blob/feat/a-training-results/reports/models/stgcn.md)。正式导出和统一评分待 B 接口。

```powershell
git clone --branch main https://github.com/MorningMartin111/SpeedCast-LA.git
cd SpeedCast-LA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-data.txt
python -m traffic.data.fixture --out data/fixture-v1
python -m traffic.data.check --data data/fixture-v1 --fixture
python -m traffic.data.download --out data/raw
python -m traffic.data.prepare --raw-dir data/raw --out data/metr-la-v1
python -m traffic.data.check --data data/metr-la-v1
```

正式数据的图规定站点顺序，前三个 ID 为 `773869,767541,767542`。下载入口校验固定 SHA-256，来源、切分、抽查值及缺失率见 [数据说明](data.md)。目前没有发布另一个团队共享数据压缩包，各成员可按命令重建。

## A0–A4 当前证据

- A0：协议、fixture、公共 Dataset、9 项数据测试与公共 PR 已完成并合并；新克隆可读取 batch。
- A1：作者三个源文件下载校验、按 ID 对齐、训练期统计、窗口、正式 bundle 与来源审计完成；独立重建指纹一致，5 项准备测试通过。
- A2：STGCN 实现及 CPU 联调、GPU 完整单轮试跑、正式种子 42 训练已完成；最多 60 epochs、早停耐心 10，实际训练 60 epochs，最佳第 58 轮验证 MAE 3.0121139473 mph。
- A3：纯时间消融保持相同其余配置与种子，正式训练 46 epochs，最佳第 36 轮验证 MAE 3.2402696340 mph；两版参数量 27276 / 23180。`temporal − STGCN` 为 0.2281556867 mph，仅是一个种子的验证观察。
- A4：入口、PR、贡献、实际配置、逐轮轻量记录、本地 checkpoint 路径及校验值已记录。最终预测包、B 统一评分及 C 拥堵分组仍待接入；未评分 test，未运行种子 43、44。

A 负责 `traffic/data/`、`traffic/models/stgcn/`、对应配置与测试、公共协议及本报告。B 的 `traffic/eval/`、DCRNN 和 C 的 STAEformer/场景地图模块由各自负责。22 项回归测试已通过；历史 CPU 证据 `stgcn-runs.json`、`stgcn-history.csv` 保留原值，新正式训练证据单独位于 [stgcn-formal-runs.json](../reports/models/stgcn-formal-runs.json)、[stgcn-formal-history.csv](../reports/models/stgcn-formal-history.csv)。完整范围、结果与命令见 [STGCN 报告](../reports/models/stgcn.md)。

## B、C 接口交接

当前 `predict --check-only` 只验证内存预测，正式验证预测预期为 `[3416,12,207]`，按数据包起点与站点顺序输出有限 mph。B 合并 `write_predictions`、`masked_scores`、`traffic.eval.run` 后，两版先导出并评分 val，交叉检查当前选择指标；配置锁定后再处理 test，并接 C 的拥堵转变分组。

权重位于 A 当前电脑的 `runs/stgcn/seed42/best.pt` 与 `runs/stgcn_temporal/seed42/best.pt`，SHA-256 在正式运行索引中。它们尚无公开下载链接；其他成员可按报告中的实际配置重建。原始数据、bundle、完整预测和权重均留在忽略目录，当前结果 PR 只交文档与轻量证据。
