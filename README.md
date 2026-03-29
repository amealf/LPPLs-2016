# LPPL Backtest

这个仓库提供一个基于 2016 年 modified profile likelihood 思路的 LPPL 回测脚本。当前主程序会沿历史滚动分析 `t2`，寻找可能进入泡沫状态、适合继续跟踪做空的候选区间。

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

## 数据读取规则

主程序先检查是否已经存在转换好的小文件：

`Data/<原始文件名去掉扩展名>__<周期>.csv`

例如：

- `Data/xagusd_30s_all__1d.csv`
- `Data/xagusd_30s_all__2h.csv`

若对应周期的小文件存在，程序直接读取它。  
若不存在，程序再读取原始文件 `Data/xagusd_30s_all.csv`，并在内存中完成重采样。

## 主程序设置

主程序顶部有几项常用设置：

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

## 生成转换文件

`extension/resample_data.py` 用来生成小体积转换文件。脚本开头已经提供默认配置：

- `DEFAULT_DATA_FILE_NAME`
- `DEFAULT_RESAMPLE_RULES`

默认周期列表示例：

```python
DEFAULT_RESAMPLE_RULES = ["1h", "2h", "day"]
```

直接运行一次即可：

```bash
python extension/resample_data.py
```

命令行参数同样可以继续使用。

示例：

```bash
python extension/resample_data.py --data-file xagusd_30s_all.csv --rules 1D 2h
```

执行后会在 `Data/` 目录中生成：

- `xagusd_30s_all__1d.csv`
- `xagusd_30s_all__2h.csv`

## 运行

```bash
python lppls_library_crash_2016_backtest.py
```

结果会写入：

`LPPL result/<程序名子文件夹>/`

输出文件名会包含品种名与样本时间范围。

## 依赖

安装依赖：

```bash
pip install -r requirements.txt
```

## GitHub 上传说明

仓库默认只保留小体积转换文件。  
原始大文件会由 `.gitignore` 跳过，例如：

`Data/xagusd_30s_all.csv`

若后续加入新的原始案例，建议继续沿用 `*_all.csv` 这一类命名方式。这样上传时会自动跳过大文件，只保留转换后的样本文件。
