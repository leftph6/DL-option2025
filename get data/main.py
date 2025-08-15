import os
import pandas as pd
from binance.client import Client
from datetime import datetime
import time

# --- 步骤 1: 设置 API 密钥 ---
# 强烈建议使用环境变量来存储您的 API 密钥，而不是直接写在代码里。
# 在您的操作系统中设置 'BINANCE_API_KEY' 和 'BINANCE_API_SECRET'。
# 如果您不想设置环境变量，也可以直接在这里填写：
# api_key = "YOUR_API_KEY"
# api_secret = "YOUR_API_SECRET"
api_key = os.getenv('BoNEmNw6zMvKEGNNXGQrzcNr9yqUGRUirE8stLvMP4tAOuPEm2cT22nf8lFtGRk48')
api_secret = os.getenv('BINANCE_API_SECRET')

# 检查 API 密钥是否存在
if not api_key or not api_secret:
    print("错误：请设置 'BINANCE_API_KEY' 和 'BINANCE_API_SECRET' 环境变量。")
    # 如果您没有API密钥，可以只使用公共API，但会有请求频率限制
    client = Client()
else:
    # 初始化客户端
    client = Client(api_key, api_secret)

# --- 步骤 2: 定义获取数据的参数 ---
symbol = 'BTCUSDT'          # 交易对
interval = Client.KLINE_INTERVAL_1HOUR  # K线间隔 (1m, 5m, 1h, 4h, 1d, 1w, 1M)
start_date = "1 Jan, 2023"  # 开始日期
end_date = "1 Jan, 2024"    # 结束日期 (币安API允许不填，默认为当前)
limit = 1000                # 每次请求返回的最大记录数 (最大为 1000)

# --- 步骤 3: 定义获取和处理数据的函数 ---
def get_all_binance_klines(symbol, interval, start_str, end_str=None):
    """
    获取指定时间范围内的所有K线数据。
    币安API的 get_historical_klines 每次最多返回1000条数据，所以需要循环获取。
    """
    # 将日期字符串转换为毫秒时间戳
    start_ts = int(datetime.strptime(start_str, '%d %b, %Y').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, '%d %b, %Y').timestamp() * 1000) if end_str else None

    all_klines = []
    current_start_ts = start_ts

    print(f"开始从 {start_str} 获取 {symbol} 的 {interval} K线数据...")

    while True:
        try:
            # 获取K线数据
            klines = client.get_historical_klines(
                symbol,
                interval,
                start_str=str(current_start_ts),
                end_str=str(end_ts) if end_ts else None,
                limit=limit
            )
            # 如果没有返回数据，则跳出循环
            if not klines:
                break

            all_klines.extend(klines)
            
            # 更新下一次请求的开始时间戳
            # K线数据的第一项是开盘时间，我们用最后一根K线的开盘时间作为下一次请求的起点
            new_start_ts = klines[-1][0] + 1

            print(f"已获取 {len(all_klines)} 条数据，最新时间: {datetime.fromtimestamp(klines[-1][0]/1000)}")

            # 如果获取的数据已经到达或超过了我们设定的结束时间，或者下一次的开始时间超过了结束时间，则停止
            if new_start_ts >= current_start_ts and (end_ts and new_start_ts >= end_ts):
                break
            
            current_start_ts = new_start_ts

            # 为了防止请求过于频繁而被API限制，可以适当暂停
            time.sleep(0.5)

        except Exception as e:
            print(f"发生错误: {e}")
            print("将在5秒后重试...")
            time.sleep(5)

    return all_klines

# --- 步骤 4: 执行数据获取并保存到CSV ---
if __name__ == '__main__':
    # 获取K线数据
    klines_data = get_all_binance_klines(symbol, interval, start_date, end_date)

    if not klines_data:
        print("未能获取到任何数据，程序退出。")
    else:
        # 将数据转换为 Pandas DataFrame
        columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        df = pd.DataFrame(klines_data, columns=columns)

        # --- 步骤 5: 数据清洗和格式化 ---
        # 将时间戳转换为可读日期格式
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

        # 将字符串类型的数值转换为浮点数或整数
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                           'Taker buy base asset volume', 'Taker buy quote asset volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)
        df['Number of trades'] = df['Number of trades'].astype(int)

        # 删除不需要的 'Ignore' 列
        df.drop('Ignore', axis=1, inplace=True)

        # --- 步骤 6: 保存为 CSV 文件 ---
        file_name = f'{symbol}_{interval}_{start_date.replace(" ", "")}_to_{end_date.replace(" ", "")}.csv'
        df.to_csv(file_name, index=False)

        print("-" * 50)
        print(f"数据获取完成！总共获取 {len(df)} 条记录。")
        print(f"数据已保存到文件: {file_name}")
        print("数据预览 (前5条):")
        print(df.head())
        print("-" * 50)