#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 上午9:56
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : logging.py
# @Software: PyCharm
"""
日志的配置
"""

import logging
import os
import time
from pathlib import Path
from typing import List

log_dir = os.path.join(os.getcwd(), 'logs', time.strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
                              "%Y-%m-%d:%H:%M:%S")

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 创建文件处理器
file_handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)


def get_logger(name: str, level: int = None) -> logging.Logger:
    """
    得到某个特定文件的日志记录器
    """
    try:
        p = Path(name)
        if p.exists():
            name = str(p.absolute().relative_to(Path.cwd()).as_posix())
    except:
        pass
    from sys import argv

    logger = root_logger.getChild(name)

    # 命令模式中DEBUG模式的标志符
    debug_flags: List[str] = ["-d", "--debug", "-vv", "-vvv" "--verbose"]

    # 控制是否进入DEBUG模式
    if level is None and any(v in argv for v in debug_flags):
        level = logging.DEBUG
    if level is None:
        level = logging.INFO

    if isinstance(level, str):
        if level.upper() == 'DEBUG':
            level = logging.DEBUG
        elif level.upper() == 'INFO':
            level = logging.INFO
        elif level.upper() == 'WARNING':
            level = logging.WARNING
        elif level.upper() == 'ERROR':
            level = logging.ERROR
    logger.setLevel(level)

    return logger
