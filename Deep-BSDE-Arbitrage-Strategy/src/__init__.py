#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep BSDE 套利策略回测系统
"""

from .deep_bsde_model import DeepBSDE, ControlNet, create_model, select_device
from .arbitrage_strategy import ArbitrageStrategy, create_strategy
from .backtest_engine import BacktestEngine, create_backtest_engine

__version__ = "1.0.0"
__author__ = "Deep BSDE Team"
__email__ = "team@deepbsde.com"

__all__ = [
    "DeepBSDE",
    "ControlNet", 
    "ArbitrageStrategy",
    "BacktestEngine",
    "create_model",
    "create_strategy",
    "create_backtest_engine",
    "select_device"
]
