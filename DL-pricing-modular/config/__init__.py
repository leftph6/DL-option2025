"""
配置管理模块初始化文件
"""

from .default_config import DEFAULT_CONFIG
from .config_manager import ConfigManager

__all__ = [
    'DEFAULT_CONFIG',
    'ConfigManager'
]
