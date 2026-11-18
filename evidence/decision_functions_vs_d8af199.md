# 决策路径函数源码比对: d8af199 vs 当前(逐函数)
## PrefetchTDRClusterGraph.splitter_node: 不同
--- 
+++ 
@@ -0,0 +1,22 @@
+def splitter_node(self, state: GraphState) -> Dict[str, Any]:
+        """覆写仅在有任务待切时生效; 返回块与基类尾部同构(基类无此缝可复用)。"""
+        tasks = state.get("tasks", deque())
+        if state.get("clusters_list") or not tasks:
+            return super().splitter_node(state)
+
+        current_task = tasks.popleft()
+        pre = self.split_prefetch.take(current_task.id)
+        if pre is None:
+            pre = self.split_prefetch.direct(current_task, state["total_queries"])
+        review_clusters, residual_clusters = pre
+
+        all_residual_clusters = list(state.get("residual_clusters", []))
+        all_residual_clusters.extend(residual_clusters)
+        print(f"   切分完成: review={len(review_clusters)}, residual={len(residual_clusters)}")
+        return {
+            "tasks": tasks,
+            "categories": state.get("categories", {}),
+            "clusters_list": review_clusters,
+            "current_cluster": None,
+            "residual_clusters": all_residual_clusters,
+        }
## TDRClusterGraph._apply_decision: 逐字相同
## TDRClusterGraph._handle_inspect_request: 逐字相同
## TDRClusterGraph._should_continue: 逐字相同
## TDRClusterGraph.dispatcher_node: 逐字相同
## TDRClusterGraph.reviewer_node: 逐字相同
## TDRClusterGraph.splitter_node: 逐字相同