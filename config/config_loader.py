"""
简化的配置管理器 - 纯配置文件模式
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def _load_config() -> Dict[str, Any]:
    """
    加载配置文件(CONFIG_PATH env 可覆盖, 支持多进程各自配置)

    Returns:
        Dict[str, Any]: 配置字典
    """
    override = os.environ.get("CONFIG_PATH")
    if override:
        with open(override, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ 配置文件加载成功: {override}")
        return config
    config_dir = Path(__file__).parent
    config_path = config_dir / "config.yaml"
    example_path = config_dir / "config.example.yaml"

    for candidate in (config_path, example_path):
        if not candidate.exists():
            continue

        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✓ 配置文件加载成功: {candidate}")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            return {}

    print(f"❌ 配置文件加载失败: 未找到 {config_path.name} 或 {example_path.name}")
    return {}


# 模块加载时直接初始化全局CONFIG
CONFIG = _load_config()
