# LPPL Backtest

这个仓库提供一个基于 2016 年 modified profile likelihood 思路的 LPPL 回测脚本,沿历史滚动分析 `t2`，寻找可能进入泡沫状态、适合继续跟踪做空的候选区间。

当前主程序是：

`lppls_library_crash_2016_backtest.py`

## 目录

- `Data/`：数据文件目录
- `extension/`：扩展模块与 resample 工具
- `LPPL result/`：运行结果输出目录
- `paper/`：参考论文

## 数据格式

程序默认读取无表头 CSV，列顺序如下：

`datetime, open, high, low, close, volume`

时间列示例：

`2025-01-01 18:00:00`


## 主程序设置

主程序顶部有几项设置：

```python
DATA_FILE_NAME = "xagusd_30s_all.csv"
START_DATE = "20241201"
END_DATE = ""
SCAN_PROFILE = "bull_year"
BACKTEST_SPEED_MODE = "fast_validation"
```

说明如下：

- `DATA_FILE_NAME`：选择 `Data/` 目录中的原始案例文件
- `START_DATE`、`END_DATE`：日期过滤，格式是 `yyyymmdd`
- `SCAN_PROFILE`：`bull_year` 适合年级别泡沫，`crash_local` 适合局部急跌
- `BACKTEST_SPEED_MODE`：`fast_validation` 适合快速研究，`full_scan` 适合细扫

## 运行

```bash
python lppls_library_crash_2016_backtest.py
```

## 依赖

安装依赖：

```bash
pip install -r requirements.txt
```
