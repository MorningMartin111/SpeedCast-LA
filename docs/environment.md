# A 的实际环境

验证日期：2026-11-21（America/Toronto）。

- Windows，Python 3.13.7。
- NumPy 2.5.3。
- PyTorch 2.14.0+cpu，`torch.version.cuda=None`，`torch.cuda.is_available()=False`。
- `.venv` 使用 `--system-site-packages` 复用本机已安装的 NumPy/PyTorch；README 为其他成员提供独立虚拟环境安装方式。

Python 3.11 是任务书的建议起点，本次实际验证使用 Python 3.13。依赖允许兼容版本；每位成员应记录自己的实际环境。

另从 GitHub 克隆 `feat/a-data-contract`，创建未继承系统包的独立环境并按 `requirements.txt` 安装，实际得到 Python 3.13.7 / NumPy 2.5.5 / PyTorch 2.15.0+cpu。fixture 生成、手算检查和全部 9 项数据测试再次通过，逻辑指纹与上面的 NumPy 2.5.3 环境完全一致。

正式数据准备另安装 `requirements-data.txt`；本机实际为 pandas 3.1.1 / PyTables 3.11.1。作者 2017 年 HDF5 的字节元数据触发 pandas 3 兼容性问题，准备脚本使用只读数值数组读取路径，保留原始文件校验值。
