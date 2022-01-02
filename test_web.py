import json
import requests
import threading
import time
import random
import numpy as np

"""
多线程测试接口的并发效率
"""

data = {
    'times': 1000,  # 并发数
    'url': 'http://localhost:1234/predict?sentence=感冒鼻子干有什么方法能快速解决'
}


def get_predict():
    global RIGHT_NUM
    global ERROR_NUM
    try:
        result = requests.get(data['url'])
        if result.status_code == 200:
            RIGHT_NUM += 1
        else:
            ERROR_NUM += 1
    except Exception as e:
        print(e)


def run():
    threads = []
    time_start = time.time()
    for i in range(data['times']):
        t = threading.Thread(target=get_predict)
        t.setDaemon(True)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time_end = time.time()
    print('============result===========')
    print("并发数：", data["times"])
    print("总耗时：", time_end - time_start)
    print("每次请求平均耗时：", (time_end - time_start) / data["times"])
    print("每秒请求数量（QPS）：", data["times"] / (time_end - time_start))
    print("正确返回：", RIGHT_NUM)
    print("错误返回：", ERROR_NUM)
    return data["times"] / (time_end - time_start)


if __name__ == "__main__":
    RIGHT_NUM = 0
    ERROR_NUM = 0
    while True:
        run()
