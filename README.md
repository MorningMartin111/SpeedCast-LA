# SpeedCast LA

ECE1553 多站点交通速度预测项目：在同一份 METR-LA 数据、时间切分和掩码约定上比较 Persistence、STGCN、DCRNN、STAEformer，并展示拥堵分布。A 负责公共数据、STGCN 和空间消融；B 负责统一评测与 DCRNN；C 负责 STAEformer、场景分组、可视化及最终报告。

公共实验约定见 [docs/protocol.md](docs/protocol.md)，A 的任务范围见 [docs/tasks/A_DATA_STGCN.md](docs/tasks/A_DATA_STGCN.md)。

从仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m traffic.data.fixture --out data/fixture-v1
python -m traffic.data.check --data data/fixture-v1 --fixture
python -m unittest discover -s tests -p "test_data.py" -v
```

Linux/macOS 激活方式为 `source .venv/bin/activate`。本机验证环境见 [docs/environment.md](docs/environment.md)。

公共读取入口：

```python
from torch.utils.data import DataLoader
from traffic.data.dataset import load_bundle

bundle = load_bundle("data/fixture-v1")
batch = next(iter(DataLoader(bundle.dataset("test"), batch_size=4)))
print(bundle.data_id, bundle.sensor_ids)
print({name: tuple(value.shape) for name, value in batch.items()})
```

fixture 固定 120 行、3 个站点，节点轴 `S02,S01,S03`，训练/验证/测试起点数 `61/1/13`。输入 `x[B,12,N,3]`，标签 `y_mph[B,12,N]`。起点 `o=96` 为 08:00，第 3 步为 08:15；Persistence 的该步有效目标数为 2，MAE 为 15.375 mph。

原始数据、生成的数据包、完整预测与模型权重通过复现命令生成，默认不提交 Git。B、C 可先接入 fixture，随后同步正式数据包的 `data_id`。

正式 METR-LA 数据已验证，可从 DCRNN 作者来源重建：

```powershell
python -m pip install -r requirements-data.txt
python -m traffic.data.download --out data/raw
python -m traffic.data.prepare --raw-dir data/raw --out data/metr-la-v1
python -m traffic.data.check --data data/metr-la-v1
```

实测 `T=34272,N=207`，train/val/test 为 `23967/3416/6844` 个起点。正式 `data_id` 为 `sha256:61a74eeffbe6f6c22d44ed1dbf02f3032ccd14ea88c09c61aa753f427e494219`。来源、原文件 SHA-256、数据质量和时间摘要见 [docs/data.md](docs/data.md)。
