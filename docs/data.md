# METR-LA 公共数据（A1）

正式数据已在 Windows / Python 3.13.7 / NumPy 2.5.3 / pandas 3.1.1 / PyTables 3.11.1 上实际准备并通过公共读取接口检查。原始车速、图和坐标均来自 [DCRNN 作者发布的数据](https://github.com/liyaguang/DCRNN#data-preparation)。上游版本固定为 `602afd9d767d3aa1c9b3eac51710d6aeee12c227`；首次获取的会话本地日期为 2026-11-21。

从项目根目录运行：

```powershell
python -m pip install -r requirements.txt -r requirements-data.txt
python -m traffic.data.download --out data/raw
python -m traffic.data.prepare --raw-dir data/raw --out data/metr-la-v1
python -m traffic.data.check --data data/metr-la-v1
python -m unittest discover -s tests -p test_prepare.py -v
```

`download` 校验下表固定的 SHA-256，不会覆盖现有文件。来源不一致会直接失败。已有原始文件可直接运行 `prepare`；若文件名不同，使用 `--speed-file`、`--graph-file`、`--locations-file` 映射。`sources.json` 记录实际来源、文件名、获取/校验日期、长度及校验值。`prepare` 会检查已有来源清单的校验值，并把来源记录写入 bundle 的 `manifest.json`。本地取得的其他版本不会被标成已验证下载。

| 原始文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `metr-la.h5` | 57038056 | `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05` |
| `adj_mx.pkl` | 680459 | `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3` |
| `graph_sensor_locations.csv` | 6341 | `eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f` |

车速文件从作者 README 链接的 [Google Drive 文件夹](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) 中的 `metr-la.h5` 获取，文件 ID 为 `1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC`。配套图和坐标通过上述固定 revision 的 `data/sensor_graph/` 获取。`traffic/data/download.py` 保存可直接重跑的完整 URL 和预期校验值。

当前电脑的原始数据位于 `data/raw/`，团队输入位于 `data/metr-la-v1/`；这两个目录不进入 Git。B、C 可以用上述命令自行生成，随后核对 `data_id`。目前没有另行发布共享压缩包。

## 实际摘要

`data_id`：`sha256:61a74eeffbe6f6c22d44ed1dbf02f3032ccd14ea88c09c61aa753f427e494219`

| 项目 | 实际值 |
| --- | --- |
| 时间点 / 站点 | 34272 / 207 |
| 日期范围 | 2012-03-01 00:00:00 至 2012-06-27 23:55:00 |
| 时间语义 | 源表无时区；保留本地钟面时间，不额外定位时区 |
| 重复时间 / 非五分钟间隔 | 0 / 0；34271 个相邻间隔均为 300 秒 |
| 切分边界 | `a=23990`、`b=27417` |
| 训练 / 验证 / 测试起点数 | **23967 / 3416 / 6844** |
| 有效速度 | 有限且严格大于 0，单位 mph |
| 总缺失率 | 0.08109350825676487 |
| 前向填补比例 | 0.08108420501856137 |
| 训练均值兜底比例 | 0.000009303238203493958 |
| 训练均值 / 总体标准差 | 58.58425765022911 / 12.822883207357794 mph |
| 缺坐标 / 缺训练参考速度的站点 | 0 / 0 |
| 日历下标范围 | `tod_slot=0…287`、`dow=0…6` |
| 首个 batch（batch size 2） | `x[2,12,207,3]`、`y_mph[2,12,207]`、`y_mask[2,12,207]` |

有向图保持原值，`adj_dir[i,j]` 代表从 `i` 到 `j`。站点顺序来自图，前 3 个是 `773869,767541,767542`。图索引 `0,103,206` 分别追溯到源速度列 `0,103,206`、站点 `773869,717488,769373`，首行速度为 `64.375,66.25,61.875` mph。准备程序显式解码 byte IDs，按 ID 重排速度列和坐标；不靠 CSV 行号对齐。

| split | 首 / 末起点 | 首起点时刻 | 首目标范围 | 末起点时刻 | 末目标范围 |
| --- | --- | --- | --- | --- | --- |
| train | 11 / 23977 | 03-01 00:55 | 03-01 01:00…01:55 | 05-23 06:05 | 05-23 06:10…07:05 |
| val | 23989 / 27404 | 05-23 07:05 | 05-23 07:10…08:05 | 06-04 03:40 | 06-04 03:45…04:40 |
| test | 27416 / 34259 | 06-04 04:40 | 06-04 04:45…05:40 | 06-27 22:55 | 06-27 23:00…23:55 |

表中日期均为 2012 年。第一个验证窗口读取 `2012-05-23 06:10…07:05` 的 12 个历史点，这些观测在发出预测时已发生；12 个未来目标全部位于验证时段。相邻 split 的未来标签时间集合已经逐项检查为互斥。每站填补只使用当前及之前的有效观测，跨真实时间缺口重置；均值、总体标准差和线性第 85 百分位参考速度只从训练时段计算。

完整每站有效率、来源追溯、窗口首末信息及样本形状在生成目录的 `audit.json` 和 `manifest.json` 中。`audit.json` 是派生报告，不参与指纹；`data_id` 覆盖规范时间、站点顺序、坐标、有向图、逻辑数组、填补、划分、scaler 以及字段语义，排除本机路径、下载日期、来源压缩容器时间。

## 兼容性与检查

作者 2017 年 HDF5 使用 byte 类型的 pandas 元数据，pandas 3 的 `read_hdf` 会产生 `TypeError`。准备模块先尝试标准读取，再对明确的单数值 block 固定格式使用只读 PyTables 兼容路径，校验列轴、数值维度、datetime64 单位和无时区语义；不修改原始文件，不读取旧频率对象的 pickle。

五项 A1 测试覆盖：乱序速度列与 byte graph IDs 对齐后指纹相同；修改测试段未来值后训练数据、scaler、参考速度及较早输入不变；旧 HDF5 byte 元数据回归；缺坐标保留站点且拒绝重复时间；拒绝图 ID 映射错误。时间缺口、填补、标准化还原、指纹与 fixture 手算检查在公共数据测试中覆盖。

这些是数据接口和预处理的实际验证结果。正式模型指标仍需 B 的统一评测入口，拥堵转变分组由 C 接入。
