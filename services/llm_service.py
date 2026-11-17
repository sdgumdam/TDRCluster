"""
LLM服务
"""
from typing import Dict, Any, Optional, List
import xml.etree.ElementTree as ET
import re

from openai import OpenAI


class LLMService:
    """LLM服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                print("❌ LLM API Key未配置")
                return

            self.client = OpenAI(
                base_url=self.config.get("base_url"),
                api_key=api_key,
                timeout=60,
            )

            print("✓ LLM客户端初始化成功")
            print(f"   模型: {self.config.get('model_name', 'unknown')}")

        except Exception as e:
            print(f"❌ LLM客户端初始化失败: {e}")
            self.client = None

    def simple_call(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 2000,
        timeout: Optional[float] = None,
    ) -> str:
        """LLM调用接口"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化")

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.get("model_name"),
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ LLM调用失败: {str(e)}")
            raise

    def _decide_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        cluster_ids: List[str],
        existing_categories: Dict[str, Any],
        allow_inspect: bool,
        max_retries: int = 3,
        log_label: str = "LLM reviewer",
    ) -> Dict[str, Any]:
        """共享重试+XML校验循环: 单簇决策调用(serial reviewer)"""
        last_result = {"valid": False, "error_message": "unknown"}
        for attempt in range(max_retries + 1):
            if attempt == 0:
                print(f"🔍 调用{log_label}...")
            else:
                print(f"🔄 {log_label}重试 {attempt}/{max_retries} ...")

            try:
                response = self.simple_call(prompt, system_message=system_prompt)
                validation_result = self._validate_xml_response(
                    response,
                    cluster_ids,
                    existing_categories,
                    allow_inspect=allow_inspect,
                )

                if validation_result["valid"]:
                    print("✅ LLM响应验证通过")
                    break

                print(f"❌ LLM响应验证失败: {validation_result['error_message']}")
                last_result = validation_result

            except Exception as e:
                print(f"❌ {log_label}调用失败: {str(e)}")
                last_result = {"valid": False, "error_message": str(e)}
        else:
            raise RuntimeError(f"LLM响应验证连续失败，最后错误: {last_result['error_message']}")

        decisions = validation_result.get("decisions", [])
        if len(decisions) != 1:
            raise RuntimeError(
                f"单簇评审期望1个决策，实际收到 {len(decisions)} 个"
            )

        return decisions[0]

    def analyze_cluster_with_retry(
        self,
        finalized_categories: List[Dict[str, Any]],
        cluster_to_review: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """评审单个micro-cluster，并返回唯一决策(serial路径)"""
        from core.prompts import create_review_prompt, create_reviewer_system_prompt
        from config.config_loader import CONFIG

        prompt = create_review_prompt(finalized_categories, cluster_to_review)
        system_prompt = create_reviewer_system_prompt(
            CONFIG.get("runtime", {}).get("taxonomy_kind", "intent")
        )
        return self._decide_with_retry(
            prompt,
            system_prompt,
            [cluster_to_review["id"]],
            {cat["id"]: cat for cat in finalized_categories},
            allow_inspect=bool(cluster_to_review.get("allow_inspect", True)),
            max_retries=max_retries,
            log_label="LLM reviewer",
        )

    def _validate_xml_response(
        self,
        xml_response: str,
        cluster_ids: List[str],
        existing_categories: Dict[str, Any] = None,
        allow_inspect: bool = True,
    ) -> Dict[str, Any]:
        """验证LLM返回的XML响应格式和完整性"""
        result = {
            "valid": False,
            "decisions": [],
            "error_message": "",
            "missing_clusters": [],
            "extra_clusters": [],
            "duplicate_clusters": [],
        }

        try:
            xml_response = xml_response.strip()
            if not xml_response.startswith("<"):
                start_pos = xml_response.find("<decisions>")
                if start_pos == -1:
                    result["error_message"] = "未找到<decisions>标签"
                    return result
                xml_response = xml_response[start_pos:]

            xml_response = re.sub(r"&(?!(?:amp|lt|gt|quot|apos);)", "&amp;", xml_response)
            root = ET.fromstring(xml_response)
            if root.tag != "decisions":
                result["error_message"] = f"根标签应为<decisions>，实际为<{root.tag}>"
                return result

            decisions = []
            decision_cluster_ids = set()

            for decision_elem in root.findall("decision"):
                id_elem = decision_elem.find("id")
                action_elem = decision_elem.find("action")

                if id_elem is None or not id_elem.text:
                    result["error_message"] = "决策缺少id字段"
                    return result
                if action_elem is None or not action_elem.text:
                    result["error_message"] = "决策缺少action字段"
                    return result

                cluster_id = id_elem.text.strip()
                if "," in cluster_id:
                    result["error_message"] = "每个决策只能引用一个micro-cluster id"
                    return result

                if cluster_id in decision_cluster_ids:
                    result["duplicate_clusters"].append(cluster_id)
                decision_cluster_ids.add(cluster_id)

                action = action_elem.text.strip().lower()

                decision = {
                    "id": cluster_id,
                    "action": action,
                }

                if action == "create":
                    name_elem = decision_elem.find("name")
                    definition_elem = decision_elem.find("definition")
                    boundary_elem = decision_elem.find("boundary_notes")

                    if name_elem is None or not name_elem.text:
                        result["error_message"] = "create决策缺少name字段"
                        return result
                    if definition_elem is None or not definition_elem.text:
                        result["error_message"] = "create决策缺少definition字段"
                        return result
                    if boundary_elem is None or not boundary_elem.text:
                        result["error_message"] = "create决策缺少boundary_notes字段"
                        return result

                    decision["name"] = name_elem.text.strip()
                    decision["definition"] = definition_elem.text.strip()
                    decision["boundary_notes"] = boundary_elem.text.strip()

                elif action == "merge":
                    target_elem = decision_elem.find("target_id")
                    name_update_elem = decision_elem.find("name_update")
                    definition_update_elem = decision_elem.find("definition_update")
                    boundary_update_elem = decision_elem.find("boundary_update")

                    if target_elem is None or not target_elem.text:
                        result["error_message"] = "merge决策缺少target_id字段"
                        return result
                    if name_update_elem is None or not name_update_elem.text:
                        result["error_message"] = "merge决策缺少name_update字段"
                        return result
                    if definition_update_elem is None or not definition_update_elem.text:
                        result["error_message"] = "merge决策缺少definition_update字段"
                        return result
                    if boundary_update_elem is None or not boundary_update_elem.text:
                        result["error_message"] = "merge决策缺少boundary_update字段"
                        return result

                    target_id = target_elem.text.strip()
                    if existing_categories is not None and target_id not in existing_categories:
                        result["error_message"] = f"merge决策的target_id '{target_id}' 不存在"
                        return result

                    decision["target_id"] = target_id
                    decision["name_update"] = name_update_elem.text.strip()
                    decision["definition_update"] = definition_update_elem.text.strip()
                    decision["boundary_update"] = boundary_update_elem.text.strip()

                elif action == "split":
                    reason_elem = decision_elem.find("reason")
                    if reason_elem is None or not reason_elem.text:
                        result["error_message"] = "split决策缺少reason字段"
                        return result
                    decision["reason"] = reason_elem.text.strip()
                elif action == "inspect":
                    if not allow_inspect:
                        result["error_message"] = "当前轮次不允许inspect决策"
                        return result
                    reason_elem = decision_elem.find("reason")
                    if reason_elem is None or not reason_elem.text:
                        result["error_message"] = "inspect决策缺少reason字段"
                        return result
                    decision["reason"] = reason_elem.text.strip()
                else:
                    result["error_message"] = f"无效的action: {action}"
                    return result

                decisions.append(decision)

            expected_cluster_ids = set(cluster_ids)
            result["missing_clusters"] = list(expected_cluster_ids - decision_cluster_ids)
            result["extra_clusters"] = list(decision_cluster_ids - expected_cluster_ids)

            if result["missing_clusters"] or result["extra_clusters"] or result["duplicate_clusters"]:
                error_parts = []
                if result["missing_clusters"]:
                    error_parts.append(f"遗漏clusters: {result['missing_clusters']}")
                if result["extra_clusters"]:
                    error_parts.append(f"多余clusters: {result['extra_clusters']}")
                if result["duplicate_clusters"]:
                    error_parts.append(f"重复clusters: {result['duplicate_clusters']}")
                result["error_message"] = "; ".join(error_parts)
                return result

            result["valid"] = True
            result["decisions"] = decisions

        except ET.ParseError as e:
            result["error_message"] = f"XML解析失败: {str(e)}"
        except Exception as e:
            result["error_message"] = f"未知错误: {str(e)}"

        return result

    def analyze_cluster_content(
        self,
        cluster_to_review: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """two_phase Phase A: 单簇孤立内容分析(无分类树输入), 返回分析字典"""
        from core.prompts import (
            create_content_analysis_prompt,
            create_content_analysis_system_prompt,
        )

        prompt = create_content_analysis_prompt(cluster_to_review)
        system_prompt = create_content_analysis_system_prompt()

        last_error = "unknown"
        for attempt in range(max_retries + 1):
            if attempt == 0:
                print(f"🔍 内容分析 {cluster_to_review.get('id')} ...")
            else:
                print(f"🔄 内容分析重试 {attempt}/{max_retries} ({cluster_to_review.get('id')}) ...")
            try:
                response = self.simple_call(prompt, system_message=system_prompt)
                analysis = self._parse_analysis_response(response)
                print(f"✅ 内容分析通过 ({cluster_to_review.get('id')}) purity={analysis.get('purity')}")
                return analysis
            except Exception as e:
                print(f"❌ 内容分析失败: {str(e)[:120]}")
                last_error = str(e)
        raise RuntimeError(f"内容分析连续失败: {last_error}")

    def _parse_analysis_response(self, xml_response: str) -> Dict[str, Any]:
        """解析并校验 <analysis> XML 响应"""
        text = xml_response.strip()
        start = text.find("<analysis>")
        end = text.find("</analysis>")
        if start == -1 or end == -1 or end < start:
            raise ValueError("未找到<analysis>标签")
        xml_text = re.sub(r"&(?!(?:amp|lt|gt|quot|apos);)", "&amp;", text[start:end + len("</analysis>")])
        root = ET.fromstring(xml_text)
        if root.tag != "analysis":
            raise ValueError(f"根标签应为<analysis>，实际为<{root.tag}>")

        def _field(name: str) -> str:
            elem = root.find(name)
            return (elem.text or "").strip() if elem is not None else ""

        analysis = {
            "summary": _field("summary"),
            "candidate_label": _field("candidate_label"),
            "candidate_definition": _field("candidate_definition"),
            "purity": _field("purity").lower(),
            "split_reason": _field("split_reason"),
        }
        if analysis["purity"] not in ("pure", "impure"):
            raise ValueError(f"purity 必须为 pure|impure，实际: {analysis['purity']!r}")
        if not analysis["summary"] or not analysis["candidate_label"]:
            raise ValueError("summary/candidate_label 不能为空")
        return analysis

