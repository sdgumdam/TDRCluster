"""
数据集管理器
"""
import json
import os
import pickle
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.config_loader import CONFIG
from .embedding_service import EmbeddingService
from .path_root import DATA_ROOT
from core.state import Query


class DatasetManager:
    """数据集管理器"""

    SUPPORTED_RAW_EXTENSIONS = (".csv", ".xlsx", ".xlsm", ".xls")

    def __init__(self):
        self.config = CONFIG
        # 数据根目录单一真源: services/path_root.DATA_ROOT (TDR_DATA_DIR 启动注入)
        self.base_path = DATA_ROOT
        print(f"📂 数据根目录: {self.base_path.resolve()}")
        self.raw_path = self.base_path / "raw_data"
        self.processed_path = self.base_path / "processed_data"

        # 确保目录存在
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        self.embedding_service = EmbeddingService()
        # 追加式断点: 各数据集已持久化到的行号(append块只写增量)
        self._persisted_upto: Dict[str, int] = {}

    def _embedding_model_stamp(self) -> Dict[str, str]:
        """当前 embedding 模型身份(换模型必须重建缓存, 防止静默混用两个向量空间)"""
        svc = self.embedding_service
        return {
            "api_style": getattr(svc, "api_style", "unknown"),
            "model_name": str(svc.config.get("model_name", "unknown")),
        }

    def _check_model_stamp(self, dataset_name: str, stamp_file) -> bool:
        """缓存存在但模型身份不符 → 视为无缓存(由 generate 重建); True=匹配或无缓存文件"""
        import json
        if not stamp_file.exists():
            return False  # 无戳=未知来源, 一律重建(旧缓存无戳文件兼容期已过)
        try:
            stamp = json.loads(stamp_file.read_text(encoding="utf-8"))
            return stamp == self._embedding_model_stamp()
        except Exception:
            return False

    def _has_embeddings(self, dataset_name: str) -> bool:
        """检查数据集是否已有embeddings(且模型身份匹配)"""
        processed_dir = self.processed_path / dataset_name
        stamp_file = processed_dir / "embedding_model.json"

        # 检查完整备份文件
        backup_file = processed_dir / f"{dataset_name}_complete.pkl"
        if backup_file.exists():
            return self._check_model_stamp(dataset_name, stamp_file)

        # 检查分离文件
        metadata_file = processed_dir / f"{dataset_name}_metadata.csv"
        embeddings_file = processed_dir / f"{dataset_name}_embeddings.npz"

        return (metadata_file.exists() and embeddings_file.exists()
                and self._check_model_stamp(dataset_name, stamp_file))

    def _resolve_raw_file(self, dataset_name: str) -> Path:
        for extension in self.SUPPORTED_RAW_EXTENSIONS:
            candidate = self.raw_path / f"{dataset_name}{extension}"
            if candidate.exists():
                return candidate

        searched = ", ".join(f"{dataset_name}{extension}" for extension in self.SUPPORTED_RAW_EXTENSIONS)
        raise FileNotFoundError(
            f"找不到原始数据文件。已尝试: {searched}"
        )

    def _load_raw_dataframe(self, raw_file: Path) -> pd.DataFrame:
        suffix = raw_file.suffix.lower()

        try:
            if suffix == ".csv":
                return pd.read_csv(raw_file)
            if suffix in {".xlsx", ".xlsm", ".xls"}:
                return pd.read_excel(raw_file)
        except ImportError as exc:
            raise RuntimeError(
                "读取Excel文件失败，请确认已安装 openpyxl（.xlsx/.xlsm）或 xlrd（.xls）。"
            ) from exc
        except Exception as exc:
            raise ValueError(f"无法读取文件 {raw_file.name}: {exc}") from exc

        raise ValueError(f"不支持的文件格式: {raw_file.suffix}")

    def _normalize_dataframe(self, df: pd.DataFrame, raw_file: Path) -> pd.DataFrame:
        normalized_columns = {
            str(column).strip().lower(): column
            for column in df.columns
        }

        if "query" not in normalized_columns:
            raise ValueError(
                f"{raw_file.name} 必须包含 `query` 列（支持 CSV / Excel，列名大小写不敏感）"
            )

        renamed = df.rename(columns={normalized_columns["query"]: "query"}).copy()
        if "category" in normalized_columns:
            renamed = renamed.rename(columns={normalized_columns["category"]: "category"})
        elif "category" not in renamed.columns:
            renamed["category"] = ""

        renamed["query"] = renamed["query"].fillna("").astype(str).str.strip()
        renamed["category"] = renamed["category"].fillna("").astype(str).str.strip()
        renamed = renamed[renamed["query"] != ""].reset_index(drop=True)

        if renamed.empty:
            raise ValueError(f"{raw_file.name} 中没有可用的 query 数据")

        return renamed

    def _append_file(self, dataset_name: str) -> Path:
        processed_dir = self.processed_path / dataset_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir / f"{dataset_name}_embedding_progress.append.pkl"

    def _progress_meta_file(self, dataset_name: str) -> Path:
        processed_dir = self.processed_path / dataset_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir / f"{dataset_name}_embedding_progress.meta.json"

    def _legacy_progress_file(self, dataset_name: str) -> Path:
        processed_dir = self.processed_path / dataset_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir / f"{dataset_name}_embedding_progress.pkl"

    def _discard_progress(self, dataset_name: str, why: str) -> Dict[str, Any]:
        print(f"   ⚠️ 断点缓存{why}, 丢弃重建")
        for f in (self._append_file(dataset_name), self._progress_meta_file(dataset_name)):
            if f.exists():
                f.unlink()
        self._persisted_upto[dataset_name] = 0
        return {"next_index": 0, "data": []}

    def _load_embedding_progress(self, dataset_name: str) -> Dict[str, Any]:
        append_file = self._append_file(dataset_name)
        meta_file = self._progress_meta_file(dataset_name)

        if append_file.exists():
            try:
                data: List[Dict[str, Any]] = []
                with open(append_file, "rb") as file:
                    while True:
                        try:
                            data.extend(pickle.load(file))
                        except EOFError:
                            break
                meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
                stamp = meta.get("model")
                if stamp is not None and stamp != self._embedding_model_stamp():
                    return self._discard_progress(dataset_name, f"来自其他模型({stamp})")
                next_index = int(meta.get("next_index", len(data)))
                # 崩溃窗口恢复(不全量丢弃): append 只在 covered(连续前缀)推进时写入,
                # data 恒为 0..len-1 连续行。meta 落后(append已fsync,meta未写)以 append 为准;
                # 尾块撕裂(pickle异常)由外层 except 兜底。
                if len(data) != next_index:
                    print(f"   ℹ️ 断点 meta({next_index})与 append({len(data)})不一致, 以 append 为准")
                    next_index = len(data)
                if data and (data[0].get("row_index", 0) != 0 or data[-1].get("row_index", -1) != len(data) - 1):
                    return self._discard_progress(dataset_name, "行序不连续(数据异常)")
                self._persisted_upto[dataset_name] = next_index
                print(f"   检测到embedding断点缓存: 已完成 {next_index} 条")
                return {"next_index": next_index, "data": data}
            except Exception as exc:
                return self._discard_progress(dataset_name, f"损坏({exc})")

        # 旧全量pkl一次性迁移到追加式(读入后改名.bak, 之后永不重写)
        legacy_file = self._legacy_progress_file(dataset_name)
        if not legacy_file.exists():
            self._persisted_upto[dataset_name] = 0
            return {"next_index": 0, "data": []}
        try:
            with open(legacy_file, "rb") as file:
                progress = pickle.load(file)
            stamp = progress.get("model")
            if stamp is not None and stamp != self._embedding_model_stamp():
                legacy_file.unlink()
                self._persisted_upto[dataset_name] = 0
                return {"next_index": 0, "data": []}
            next_index = int(progress.get("next_index", 0))
            data = progress.get("data", [])
            if next_index != len(data):
                raise ValueError(f"next_index={next_index} != len(data)={len(data)}")
            with open(append_file, "wb") as file:
                pickle.dump(data, file, protocol=4)
                file.flush()
                os.fsync(file.fileno())
            meta_file.write_text(json.dumps({
                "model": self._embedding_model_stamp(),
                "next_index": next_index,
            }, ensure_ascii=False), encoding="utf-8")
            os.replace(legacy_file, legacy_file.with_suffix(".pkl.migrated"))
            self._persisted_upto[dataset_name] = next_index
            print(f"   旧断点pkl已迁移为追加式: 已完成 {next_index} 条")
            return {"next_index": next_index, "data": data}
        except Exception as exc:
            print(f"   ⚠️ 旧断点pkl损坏({exc}), 丢弃重建")
            legacy_file.unlink(missing_ok=True)
            self._persisted_upto[dataset_name] = 0
            return {"next_index": 0, "data": []}

    def _save_embedding_progress(self, dataset_name: str, next_index: int, data: List[Dict[str, Any]]):
        """追加式: 把行号落在 [persisted, next_index) 的未持久化记录追加进 append 文件;
        按行号筛选(非列表位置) — 续跑时 data 只含本 run 行, 位置与行号错位"""
        persisted = self._persisted_upto.get(dataset_name, 0)
        if next_index > persisted:
            chunk = [r for r in data if persisted <= r.get("row_index", 0) < next_index]
            if chunk:
                with open(self._append_file(dataset_name), "ab") as file:
                    pickle.dump(chunk, file, protocol=4)
                    file.flush()
                    os.fsync(file.fileno())
            self._persisted_upto[dataset_name] = next_index
        meta_file = self._progress_meta_file(dataset_name)
        tmp_file = meta_file.with_suffix(".json.tmp")
        tmp_file.write_text(json.dumps({
            "model": self._embedding_model_stamp(),
            "next_index": next_index,
        }, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_file, meta_file)

    def _clear_embedding_progress(self, dataset_name: str):
        for f in (
            self._append_file(dataset_name),
            self._progress_meta_file(dataset_name),
            self._legacy_progress_file(dataset_name),
        ):
            if f.exists():
                f.unlink()
        self._persisted_upto[dataset_name] = 0

    def generate_embeddings(self, dataset_name: str, force_regenerate: bool = False) -> str:
        """
        为数据集生成embeddings

        Args:
            dataset_name: 数据集名称
            force_regenerate: 是否强制重新生成

        Returns:
            str: 处理后数据的保存路径
        """
        # 检查是否已存在且不强制重新生成
        if not force_regenerate and self._has_embeddings(dataset_name):
            print(f"✓ 数据集 {dataset_name} 已有embeddings，直接复用")
            return str(self.processed_path / dataset_name)

        # 检查原始文件
        raw_file = self._resolve_raw_file(dataset_name)

        print(f"🔄 开始为数据集 {dataset_name} 生成embeddings...")
        print(f"   原始文件: {raw_file}")

        # 读取原始数据
        df = self._normalize_dataframe(self._load_raw_dataframe(raw_file), raw_file)
        print(f"   读取到 {len(df)} 条记录")

        # 准备数据
        embedding_service = self.embedding_service

        print("   开始生成embeddings...")

        legacy_file = self._legacy_progress_file(dataset_name)
        if force_regenerate and (legacy_file.exists() or self._append_file(dataset_name).exists()):
            self._clear_embedding_progress(dataset_name)

        progress = self._load_embedding_progress(dataset_name)

        # 批量处理
        batch_size = 10
        data_for_embedding = progress["data"]
        start_index = progress["next_index"]
        # 旧进度文件无 row_index 或越界: 只保留 next_index 之前的行并按行序排
        if data_for_embedding and "row_index" not in data_for_embedding[0]:
            data_for_embedding = data_for_embedding[:start_index]
        else:
            data_for_embedding = sorted(
                (r for r in data_for_embedding if r.get("row_index", 0) < start_index),
                key=lambda r: r["row_index"],
            )

        batch_bounds = [(bs, min(bs + batch_size, len(df))) for bs in range(start_index, len(df), batch_size)]
        max_workers = int(self.config.get("embedding", {}).get("max_concurrency", 8) or 8)
        lock = threading.Lock()
        failed_batches: list = []
        # 并发完成序 != 行序: 按 batch_start 暂存, 由主线程按行序装配, 杜绝错位
        batch_results: dict = {}

        def _embed_batch(batch_range):
            batch_start, batch_end = batch_range
            batch_df = df.iloc[batch_start:batch_end]
            try:
                texts = [str(row["query"]) for _, row in batch_df.iterrows()]
                embeddings = embedding_service.get_embeddings(texts)
                assembled = [
                    {
                        "row_index": row_index,
                        "query": row["query"],
                        "category": row.get("category", ""),
                        "embedding": embedding,
                    }
                    for row_index, (_, row), embedding in zip(
                        range(batch_start, batch_end), batch_df.iterrows(), embeddings
                    )
                ]
                with lock:
                    batch_results[batch_start] = assembled
                    # 进度只推进到从 start_index 起连续成功的位置(失败批次后不前移, 续跑重试)
                    covered = start_index
                    for bs, be in batch_bounds:
                        if bs in batch_results:
                            covered = be
                        else:
                            break
                    merged = [r for bs, _ in batch_bounds if bs in batch_results for r in batch_results[bs]]
                    merged.sort(key=lambda r: r["row_index"])
                    self._save_embedding_progress(dataset_name, covered, merged)
                    print(f"   进度: {covered}/{len(df)} (批次处理)")
            except Exception as e:
                with lock:
                    failed_batches.append((batch_start, batch_end, str(e)[:160]))
                print(f"   警告: 批次 {batch_start}-{batch_end} 失败，错误: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_embed_batch, batch_bounds))

        # 主线程按行序终装配: 续跑前缀 + 本 run 成功批次(缺前缀会把断点续跑判成低成功率)
        resume_prefix = [
            r for r in data_for_embedding if r.get("row_index", 0) < start_index
        ]
        data_for_embedding = resume_prefix + [
            r for bs, _ in batch_bounds if bs in batch_results for r in batch_results[bs]
        ]
        data_for_embedding.sort(key=lambda r: r["row_index"])
        covered = start_index
        for bs, be in batch_bounds:
            if bs in batch_results:
                covered = be
            else:
                break
        if covered > start_index:
            self._save_embedding_progress(dataset_name, covered, data_for_embedding)

        if failed_batches:
            for bs, be, msg in failed_batches[:5]:
                print(f"   失败批次明细: {bs}-{be}: {msg}")
            print(f"   共 {len(failed_batches)} 个批次失败")

        # 验证生成的数据完整性
        if len(data_for_embedding) == 0:
            raise RuntimeError("未成功生成任何embeddings，不保存缓存文件")

        expected_count = len(df)
        actual_count = len(data_for_embedding)
        success_rate = actual_count / expected_count

        print(f"   Embedding生成统计:")
        print(f"     预期: {expected_count} 条")
        print(f"     实际: {actual_count} 条")
        print(f"     成功率: {success_rate:.2%}")

        # 如果成功率太低，不保存缓存
        if success_rate < 0.9:  # 成功率低于90%不保存
            raise RuntimeError(f"Embedding生成成功率过低 ({success_rate:.2%})，不保存缓存文件")

        # 保存数据
        saved_path = embedding_service.save_embeddings(data_for_embedding, dataset_name)
        self._clear_embedding_progress(dataset_name)
        print(f"✓ Embeddings已保存到: {saved_path}")

        return saved_path

    def load_dataset_as_queries(self, dataset_name: str, sample_size: Optional[int] = None) -> List[Query]:
        """
        加载数据集并转换为Query对象

        Args:
            dataset_name: 数据集名称
            sample_size: 采样大小（可选）

        Returns:
            List[Query]: Query对象列表
        """
        # 确保数据集有embeddings
        if not self._has_embeddings(dataset_name):
            print(f"⚠️ 数据集 {dataset_name} 没有embeddings，开始生成...")
            self.generate_embeddings(dataset_name)
        
        # 加载数据
        embedding_service = self.embedding_service
        data = embedding_service.load_embeddings(dataset_name)

        print(f"📂 成功加载数据集 {dataset_name}: {len(data)} 条记录")

        # 采样（如果需要）
        if sample_size and sample_size < len(data):
            import random

            data = random.sample(data, sample_size)
            print(f"   随机采样: {sample_size} 条记录")

        # 转换为Query对象
        queries = []
        for i, item in enumerate(data):
            query = Query(
                id=f"{dataset_name}-{i:04d}",
                content=item["query"],
                embedding=item["embedding"],
            )
            queries.append(query)

        print(f"✓ 准备完成 {len(queries)} 个Query对象")
        return queries
