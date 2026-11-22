# 公共数据接口验收

2026-11-21，Windows / Python 3.13.7 / NumPy 2.5.3 / PyTorch 2.14.0+cpu。

仓库根目录执行：

```powershell
python -m traffic.data.fixture --out data/fixture-v1
python -m traffic.data.check --data data/fixture-v1 --fixture
python -m unittest discover -s tests -p "test_data.py" -v
```

实际 fixture 指纹：`sha256:d4e4819ec1091324aa573674d513309b029c2530c427f3def0b77c5142e00739`。

- 时间范围 `2012-03-05 00:00:00` 至 `09:55:00`，120 行。
- 节点顺序 `S02,S01,S03`；有向边 `S02→S01=1`，`S01→S03=0.5`。
- 时间边界 `84,96`；train/val/test 起点数 `61/1/13`。
- `o=96` 为 `08:00`，第三步为 `08:15`。
- 该步 Persistence 为 `[64,60,50]`，原始目标为 `[64.75,缺测,20]`；S01 标签掩码为假，起点处也缺测，但历史前向填充值仍为 60。
- 有效目标 `n=2`，MAE `15.375 mph`，RMSE `21.219831526192664 mph`，bias `14.625 mph`。
- S03 训练期参考速度 `50 mph`，此目标为 `20 mph`。
- `DataLoader(batch_size=4)` 读取 test 的 `x=(4,12,3,3)`、`y_mph=(4,12,3)`、`y_mask=(4,12,3)`、`last_mph=(4,3)`、`origin_observed=(4,3)`、`origin_idx=(4,)`。

这些是模拟数据的接口检查。正式数据性能与 B 的公共评测结果另行记录。
