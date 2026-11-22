# A 的实际环境

验证日期：2026-11-21（America/Toronto）。

- Windows，Python 3.13.7。
- NumPy 2.5.3。
- PyTorch 2.14.0+cpu，`torch.version.cuda=None`，`torch.cuda.is_available()=False`。
- `.venv` 使用 `--system-site-packages` 复用本机已安装的 NumPy/PyTorch；README 为其他成员提供独立虚拟环境安装方式。

Python 3.11 是任务书的建议起点，本次实际验证使用 Python 3.13。依赖允许兼容版本；每位成员应记录自己的实际环境。
