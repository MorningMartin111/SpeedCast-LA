# A：STGCN 与空间消融

## 实现与引用

根据 [Yu、Yin、Zhu，IJCAI 2018](https://www.ijcai.org/proceedings/2018/0505.pdf) 重新实现 PyTorch STGCN，参考 [作者代码](https://github.com/VeritasYin/STGCN_IJCAI-18)，上游 revision `4ab75ab69af354d6a0ab86aefa082d389b187127`。源码与权重均为本项目重新编写/训练，未复制上游代码。作者论文使用 PeMSD7 数据，报告 15/30/45 分钟任务；本项目使用 METR-LA、12 个五分钟未来值，不能直接把论文数值当作本项目结果。

输入 `x[B,12,N,3]` 的通道依次为标准化填补速度、`tod_slot/288`、浮点形式的整数星期。两个 ST 块各由时间 GLU、空间层和时间 GLU 构成，时间核 3，隐藏宽度 32，dropout 0.1。有效卷积后的时间长度为 `12→10→8→6→4`。每站独立共享的线性头将 4×32 特征直接映射到 12 个标准化未来速度，输出 `[B,12,N]`，随后用公共 scaler 还原 mph。该多步输出头与按单站特征通道的 LayerNorm 为项目选择。

公共有向图在数据包中原样保留。模型内只执行一次 `A_sym=max(A,Aᵀ)`、去自环、`L=I-D^(-1/2)A_symD^(-1/2)`，固定 `lambda_max=2` 得 `L_scaled=L-I`；Chebyshev 阶数 3，使用 `T0,T1,T2`。零度节点的逆平方根度置 0，`L_ii=1`，保证有限值。图支持作为模型 buffer 保存。

`stgcn_temporal` 只把图卷积换成各站独立共享的 1×1 通道投影；时间卷积、输出头、宽度、归一化、dropout、数据顺序、种子和训练设置相同。时间卷积的节点宽度为 1，LayerNorm 仅沿单站特征通道，预测头不混合站点。参数量分别为 **27,276** 和 **23,180**，参数量差异需与消融结果一起报告。

## 正式种子 42 训练结果

两版已从随机权重完成同一预算的完整 METR-LA 训练：Adam lr=0.001、batch size=32、最多 60 epochs、早停耐心 10、种子 42，未设置训练或验证 batch 上限。训练顺序使用独立固定生成器；模型按全部有效验证标量的 `all12` MAE 选择最佳 checkpoint，再重载复算。两版在同一 GPU 上顺序执行。

数据指纹为 `sha256:61a74eeffbe6f6c22d44ed1dbf02f3032ccd14ea88c09c61aa753f427e494219`，时间点/站点 `34272/207`，起点数 `23967/3416/6844`。每个训练 epoch 实际覆盖 55289280 个有效目标；每次完整验证覆盖 7892981 个有效目标。两版均无全缺测 batch 跳过。

| 模型 | 种子 | 参数量 | 实际 epochs | 最佳 epoch | 最佳验证 all12 MAE（mph） | 训练及恢复耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stgcn | 42 | 27276 | 60 | 58 | 3.0121139473 | 591.5306 |
| stgcn_temporal | 42 | 23180 | 46 | 36 | 3.2402696340 | 333.0815 |

STGCN 达到 60 轮上限；temporal 在最佳第 36 轮后连续 10 次验证未改善，于第 46 轮早停。两版分别用新建模型独立重载最佳 checkpoint，完整验证 MAE 均精确复现；公共推理入口的验证检查均为 `[3416,12,207]`，mph 数值有限且起点、站点顺序一致。

本次 `temporal − STGCN` 的验证 MAE 差值为 **0.2281556867 mph**，完整版的验证误差较低。这是相同训练设置下**一个种子**的实际观察；两版参数量不同（27276 / 23180），不能据此给出多种子稳定收益或严格等参数容量的因果结论。尚未经 B 的统一评分器交叉检查，也未查看或评分正式 test。

正式训练基于干净提交 `9fae1063de461298e9ff8e78409fe4d9780d78c8`，两版 `code_dirty=false`。GPU 环境为 Python 3.13.7 / NumPy 2.5.3 / PyTorch 2.11.0+cu128 / CUDA 12.8；设备为 NVIDIA GeForce RTX 4060 Laptop GPU。上表时间取实际 `run.json.training_seconds`，包含训练、逐轮验证、checkpoint 保存重载及最佳权重复验；逐轮 CSV 的 `elapsed_seconds` 是累计时间。早停会使实际 epoch 数不同，因此耗时不是同 epoch 吞吐基准。

运行索引、实际生效配置、环境、完整来源元数据、checkpoint/配置/日志 SHA-256 和本地路径见 [stgcn-formal-runs.json](stgcn-formal-runs.json)，真实逐轮记录见 [stgcn-formal-history.csv](stgcn-formal-history.csv)。本机权重分别为 `runs/stgcn/seed42/best.pt` 与 `runs/stgcn_temporal/seed42/best.pt`；尚未发布公共下载地址。Git 中保存轻量证据，原始数据、bundle、权重和完整预测数组继续留在忽略目录。

## GPU 单轮试跑与历史 CPU 联调

GPU 单轮试跑独立保存在 `runs/pilot/stgcn/seed42-e1-r01/`，覆盖完整训练集与验证集，验证 MAE 为 **3.3784245613 mph**。同步计时的单轮训练加验证为 **10.0722686000 秒**；含 checkpoint 重载复验为 10.6282766000 秒。该试跑仅确认设备、数据覆盖与时间预算，未混入正式 60 epoch 上限的两版比较。

此前 CPU 联调证据保留原值，索引 [stgcn-runs.json](stgcn-runs.json) 和逐轮记录 [stgcn-history.csv](stgcn-history.csv) 未修改。它们来自干净提交 `a236259c465afd0ceb2585ca98c0b9fcb2686cd2`，当时使用 Python 3.13.7 / NumPy 2.5.3 / PyTorch 2.14.0+cpu，另一个独立环境为 NumPy 2.5.5 / PyTorch 2.15.0+cpu。

| 联调数据 / 预算 | 模型 | 实际验证 MAE（mph） | 有效验证标量 | 训练及恢复耗时（秒） |
| --- | --- | ---: | ---: | ---: |
| fixture，1 epoch | STGCN | 8.3201096853 | 36 | 0.0456 |
| fixture，1 epoch | temporal | 8.4585919910 | 36 | 0.0546 |
| METR-LA，仅训练 1 batch | STGCN | 10.1456201126 | 7892981 | 18.5250 |
| METR-LA，仅训练 1 batch | temporal | 9.2955299313 | 7892981 | 7.5097 |

上述 CPU 运行是接口联调，不用于正式性能排名；两版曾并行运行，耗时不作计算性能比较。公共数据、正式准备和模型共 22 项回归测试通过。模型测试覆盖前后向、参数更新、小样本损失下降、eval 重载一致、拉普拉斯/零度节点、有效标量加权 MAE、全缺测 batch 与运行元数据。局部依赖测试只修改另一节点历史：纯时间版目标预测不变，完整版在该构造样本中发生变化；该检查本身不证明真实空间收益。

## 重现命令

从已合并的团队 `main` 克隆或同步，在项目环境中运行。GPU 命令要求环境中的 PyTorch 已支持本机 CUDA；目录非空时请使用新的 run 路径。

```powershell
python -m pip install -r requirements.txt -r requirements-data.txt
python -m traffic.data.download --out data/raw
python -m traffic.data.prepare --raw-dir data/raw --out data/metr-la-v1
python -m traffic.data.check --data data/metr-la-v1
python -m traffic.models.stgcn.train --data data/metr-la-v1 --config configs/stgcn.json --seed 42 --run-dir runs/stgcn/seed42 --device cuda
python -m traffic.models.stgcn.train --data data/metr-la-v1 --config configs/stgcn_temporal.json --seed 42 --run-dir runs/stgcn_temporal/seed42 --device cuda
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split val --device cuda --check-only
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn_temporal/seed42 --split val --device cuda --check-only
```

首次 CPU 联调可先生成 fixture，再把 `--data` 切为 `data/fixture-v1`、`--device cpu --max-epochs 1`、run 目录切为 `runs/fixture/<model>/seed42`。正式单批联调另加 `--max-train-batches 1` 并使用 `runs/smoke/<model>/seed42`；这些覆盖不得用于正式结果比较。

## 下一项团队依赖

B 的 `traffic.eval.io.write_predictions`、`traffic.eval.metrics.masked_scores` 和 `traffic.eval.run` 尚未加入仓库。当前 MAE 按协议对有效标量汇总，B 的评分函数接入后会自动交叉检查；`--check-only` 只验证内存预测，不写最终预测包。B 合并公共接口后，对两版分别导出和评分 val：

```powershell
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split val --device cuda
python -m traffic.eval.run --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split val
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn_temporal/seed42 --split val --device cuda
python -m traffic.eval.run --data data/metr-la-v1 --run-dir runs/stgcn_temporal/seed42 --split val
```

配置固定后，再为两版将 `--split val` 改为 `--split test`，并接入 C 的拥堵转变分组。当前没有正式 `val.npz` / `test.npz` 或 `metrics_val.csv` / `metrics_test.csv`，没有 test 结果或多种子结论。种子 43、44 尚未执行。
