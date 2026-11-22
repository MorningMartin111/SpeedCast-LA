# A 的交接与个人贡献

公开仓库：[MorningMartin111/SpeedCast-LA](https://github.com/MorningMartin111/SpeedCast-LA)。最初工作目录为空，A 在当前 GitHub 账户创建仓库，使用本地仓库级 Git 身份提交本人负责部分。

## PR 与当前分支

| 顺序 | PR | 工作分支 | 内容 |
| --- | --- | --- | --- |
| 1 | [公共协议、fixture、读取接口](https://github.com/MorningMartin111/SpeedCast-LA/pull/1) | `feat/a-data-contract` | 可独立合并；共同起始 commit `7b281b4efb68de5bdc8a555625dec783dd989ef1` |
| 2 | [正式 METR-LA 数据](https://github.com/MorningMartin111/SpeedCast-LA/pull/2) | `feat/a-real-data` | 基于 PR #1；来源校验、准备脚本、正式审计 |
| 3 | [STGCN 与空间消融](https://github.com/MorningMartin111/SpeedCast-LA/pull/3) | `feat/a-stgcn` | 基于 PR #2；模型、训练/推理、验收和报告 |

三个 PR 保持待审阅。依次合并 #1、#2、#3；合并前一项后将下一项的 base 改成 `main`。当前工作分支为 `feat/a-stgcn`，`main` 暂只有初始任务文档；提前联调应显式使用对应功能分支。

## 可直接发给 B、C 的第一条消息

> B、C：公共约定、小样本和读取接口已经跑通，PR：https://github.com/MorningMartin111/SpeedCast-LA/pull/1 。共同起始 commit：`7b281b4`，分支 `feat/a-data-contract`。`load_bundle` 位于 `traffic/data/dataset.py`，最短 batch 示例见 README；不需要先下载 METR-LA。下面从新克隆运行，预期起点数 `61/1/13`、站点顺序 `S02,S01,S03`。`o=96` 为 08:00，第三步是 08:15；S01 缺测掩码为假，S03 从 50 降到 20，手算 MAE 为 15.375 mph。数据回归、干净克隆和独立环境均已通过。

```powershell
git clone --branch feat/a-data-contract https://github.com/MorningMartin111/SpeedCast-LA.git
cd SpeedCast-LA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m traffic.data.fixture --out data/fixture-v1
python -m traffic.data.check --data data/fixture-v1 --fixture
```

已有自己分支的成员可以先 `git fetch origin`，再把 `origin/feat/a-data-contract` 合入自己的开发分支。PR #1 合并后直接同步团队 `main`。

## 正式数据的第二条消息

> 正式 METR-LA 已准备，数据 PR：https://github.com/MorningMartin111/SpeedCast-LA/pull/2 ，分支 `feat/a-real-data`。正式 `data_id=sha256:61a74eeffbe6f6c22d44ed1dbf02f3032ccd14ea88c09c61aa753f427e494219`，`T=34272,N=207`，起点数 `23967/3416/6844`；图规定站点顺序，前三个 ID 为 `773869,767541,767542`。下面命令下载作者原文件并核对固定 SHA-256，输出 `data/metr-la-v1/`。请核对同一指纹后再开始正式训练。来源、时间边界、抽查值和缺失率见 `docs/data.md`。当前未另行发布共享压缩包。

```powershell
python -m pip install -r requirements-data.txt
python -m traffic.data.download --out data/raw
python -m traffic.data.prepare --raw-dir data/raw --out data/metr-la-v1
python -m traffic.data.check --data data/metr-la-v1
```

## A0–A4 当前证据与下一步

- A0：公共协议原样落库、fixture、Dataset、9 项数据测试、独立公共 PR 已完成；GitHub 新克隆可读取 batch。
- A1：作者三个原始文件下载及校验、按 ID 对齐、训练期统计、时间窗口、正式 bundle 与来源审计完成。另一目录及独立依赖环境重建指纹相同，5 项准备测试通过。
- A2：STGCN、训练与预测入口完成，CPU 前后向、损失下降、权重重载、fixture 单轮、正式单批及完整验证推理完成；尚未执行 60 epoch 完整预算实验。
- A3：纯时间消融与相同配置、种子、样本顺序完成，跨节点依赖检查及参数量/耗时已记录；空间收益须等同预算正式训练及统一评分后才可判断。
- A4：入口、PR、个人贡献、命令、轻量日志、checkpoint 校验值、局限及交接均已记录；正式预测文件、B 统一指标和 C 拥堵分组结果待接入。

A 本次提交 `traffic/data/`、`traffic/models/stgcn/`、对应配置与测试、公共协议及 A 的报告。未实现 B 的 `traffic/eval/` 或 C 的场景/地图模块。已完成的检查总计 22 项；正式数据与模型联调结果由实际日志生成，见 `reports/data/metr-la-v1.json`、`reports/models/stgcn-runs.json` 和 `reports/models/stgcn-history.csv`。

B 的模块缺席时继续使用 `predict --check-only`，该命令验证内存预测且不生成最终预测包。B 合并 `write_predictions`、`masked_scores` 与 `traffic.eval.run` 后，两版分别执行 val 导出及评分；配置固定后再导出和评分 test，并接 C 的拥堵转变分组。完整实验与接入命令见 [STGCN 报告](../reports/models/stgcn.md)。权重当前仅存在本机，均为联调权重；正式训练完成后再发布团队可访问的权重及校验值。
