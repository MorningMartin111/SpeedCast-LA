# A：STGCN 与空间消融

## 实现与引用

根据 [Yu、Yin、Zhu，IJCAI 2018](https://www.ijcai.org/proceedings/2018/0505.pdf) 重新实现 PyTorch STGCN，参考 [作者代码](https://github.com/VeritasYin/STGCN_IJCAI-18)，上游 revision `4ab75ab69af354d6a0ab86aefa082d389b187127`。源码与权重均为本项目重新编写/训练，未复制上游代码。作者论文使用 PeMSD7 数据，报告 15/30/45 分钟任务；本项目使用 METR-LA、12 个五分钟未来值，不能直接把论文数值当作本项目结果。

输入 `x[B,12,N,3]` 的通道依次为标准化填补速度、`tod_slot/288`、浮点形式的整数星期。两个 ST 块各由时间 GLU、空间层和时间 GLU 构成，时间核 3，隐藏宽度 32，dropout 0.1。有效卷积后的时间长度为 `12→10→8→6→4`。每站独立共享的线性头将 4×32 特征直接映射到 12 个标准化未来速度，输出 `[B,12,N]`，随后用公共 scaler 还原 mph。该多步输出头与按单站特征通道的 LayerNorm 为项目选择。

公共有向图在数据包中原样保留。模型内只执行一次 `A_sym=max(A,Aᵀ)`、去自环、`L=I-D^(-1/2)A_symD^(-1/2)`，固定 `lambda_max=2` 得 `L_scaled=L-I`；Chebyshev 阶数 3，使用 `T0,T1,T2`。零度节点的逆平方根度置 0，`L_ii=1`，保证有限值。图支持作为模型 buffer 保存。

`stgcn_temporal` 只把图卷积换成各站独立共享的 1×1 通道投影；时间卷积、输出头、宽度、归一化、dropout、数据顺序、种子和训练设置相同。时间卷积的节点宽度为 1，LayerNorm 仅沿单站特征通道，预测头不混合站点。参数量分别为 **27,276** 和 **23,180**，参数量差异需与消融结果一起报告。

## 已验证与适用范围

2026-11-21，本机 Python 3.13.7 / NumPy 2.5.3 / PyTorch 2.14.0+cpu；另一个独立环境为 NumPy 2.5.5 / PyTorch 2.15.0+cpu。全部 **22 项测试通过**（公共数据 9、正式准备 5、模型 8）。无 CUDA，GPU 路径尚未实测。

模型测试覆盖两版前向/反向与参数更新、小样本损失下降、eval 权重保存重载一致、拉普拉斯缩放/零度节点、有效标量加权验证 MAE、全缺测 batch 跳过和运行元数据。局部依赖测试在 eval 模式仅改变另一节点历史：纯时间版目标预测不变，完整版在该构造样本中发生变化；此检查不能推出真实数据上的空间收益。

fixture 两版各完成一次 CPU 训练及验证。正式 METR-LA 两版也完成单批训练、checkpoint 恢复和完整验证集的内存推理检查。正式数据输入为 `[B,12,207,3]`，完整验证预测为 `[3416,12,207]`，按数据包起点和站点顺序输出有限 mph。

交付运行均来自干净工作树的提交 `a236259c465afd0ceb2585ca98c0b9fcb2686cd2`，`code_dirty=false`。完整运行索引、checkpoint / 日志 / 源码校验值及实际配置见 [stgcn-runs.json](stgcn-runs.json)，逐轮记录见 [stgcn-history.csv](stgcn-history.csv)。权重目前仅在该电脑的相对 run 路径中，其他成员可用下面命令重建；尚未发布团队共享权重。

| 联调数据 / 预算 | 模型 | 实际验证 MAE（mph） | 有效验证标量 | 训练及恢复耗时（秒） |
| --- | --- | ---: | ---: | ---: |
| fixture，1 epoch | STGCN | 8.3201096853 | 36 | 0.0456 |
| fixture，1 epoch | temporal | 8.4585919910 | 36 | 0.0546 |
| METR-LA，仅训练 1 batch | STGCN | 10.1456201126 | 7892981 | 18.5250 |
| METR-LA，仅训练 1 batch | temporal | 9.2955299313 | 7892981 | 7.5097 |

METR-LA 两版使用相同训练 batch（78178 个有效目标）、完整验证集和种子 42。以上是联调观察，B 的评分交叉检查待执行，不能用于正式性能排名或空间收益结论。两版在同机并行运行，耗时包含验证及重载复验，不能当作模型计算性能基准。未查看正式 test 指标。

训练使用有效原始标签的 MAE（mph），验证按所有有效标量汇总 `all12` MAE，保存并重载最佳 checkpoint。训练样本的 shuffle 使用独立且固定的生成器；两版相同种子保持相同样本顺序。默认 Adam lr=0.001、batch size=32、最多 60 epochs、早停耐心 10、种子 42，全部有效覆盖写入各 run 的 `config.json`。`run.json` 保存提交、工作树状态、数据指纹、环境、源码校验值、最佳 epoch、权重及日志 SHA-256 和耗时。

## 重现命令

仓库根目录、项目虚拟环境下运行；目录非空时需改用新的 run 路径。

```powershell
python -m traffic.data.fixture --out data/fixture-v1
python -m traffic.models.stgcn.train --data data/fixture-v1 --config configs/stgcn.json --seed 42 --run-dir runs/fixture/stgcn/seed42 --device cpu --max-epochs 1
python -m traffic.models.stgcn.predict --data data/fixture-v1 --run-dir runs/fixture/stgcn/seed42 --split val --check-only
python -m traffic.models.stgcn.train --data data/fixture-v1 --config configs/stgcn_temporal.json --seed 42 --run-dir runs/fixture/stgcn_temporal/seed42 --device cpu --max-epochs 1
python -m traffic.models.stgcn.predict --data data/fixture-v1 --run-dir runs/fixture/stgcn_temporal/seed42 --split val --check-only
python -m traffic.models.stgcn.train --data data/metr-la-v1 --config configs/stgcn.json --seed 42 --run-dir runs/smoke/stgcn/seed42 --device cpu --max-epochs 1 --max-train-batches 1
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/smoke/stgcn/seed42 --split val --check-only
python -m traffic.models.stgcn.train --data data/metr-la-v1 --config configs/stgcn_temporal.json --seed 42 --run-dir runs/smoke/stgcn_temporal/seed42 --device cpu --max-epochs 1 --max-train-batches 1
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/smoke/stgcn_temporal/seed42 --split val --check-only
```

正式实验入口（已实现，完整预算实验尚未执行）：

```powershell
python -m traffic.models.stgcn.train --data data/metr-la-v1 --config configs/stgcn.json --seed 42 --run-dir runs/stgcn/seed42
python -m traffic.models.stgcn.train --data data/metr-la-v1 --config configs/stgcn_temporal.json --seed 42 --run-dir runs/stgcn_temporal/seed42
```

## 下一项团队依赖

B 的 `traffic.eval.io.write_predictions` 和 `traffic.eval.metrics.masked_scores` 尚未加入仓库。当前训练使用符合协议的有效标量 MAE 汇总；B 的评分函数出现后会自动交叉检查。`--check-only` 只验证内存预测，正式导出严格调用 B 的公共写入函数。

待 B 的公共接口合并后，为两版分别导出并评分 val；配置锁定后再处理 test。以下命令对 `stgcn_temporal` 同样执行：

```powershell
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split val
python -m traffic.eval.run --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split val
python -m traffic.models.stgcn.predict --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split test
python -m traffic.eval.run --data data/metr-la-v1 --run-dir runs/stgcn/seed42 --split test
```

C 的拥堵转变分组也待接入。当前没有发布正式 `metrics_val.csv` / `metrics_test.csv`，没有关于 METR-LA 空间收益的结论；单批训练结果仅用于联调。后续先完成同一预算的种子 42，再视资源补充 43、44。
