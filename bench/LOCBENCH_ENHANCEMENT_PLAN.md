# codexlens-search LocBench 增强计划

## 现状

| 指标 | codexlens-search (3实例) | LocAgent (论文) |
|------|------------------------|----------------|
| File Recall@1 | 16.7% | ~70% |
| File Recall@5 | 33.3% | ~92.7% |
| Function Recall@10 | 16.7% | ~60%+ |

## 三类未命中模式

| # | 模式 | 示例 | 占比(估) | 需要的能力 |
|---|------|------|---------|-----------|
| 1 | **间接依赖** | 问题说优化A函数，但修复需要同时改A调用的B函数 | ~40% | 依赖图遍历 |
| 2 | **症状-根因鸿沟** | 用户描述"标签不显示"，根因是底层segment2box函数 | ~35% | LLM推理 |
| 3 | **超短查询** | "CORS should not be * by default" 一句话 | ~25% | 查询扩展 |

---

## Phase 1: 搜索增强 (无LLM依赖)

**目标**: File Recall@5 → 45-50%
**解决**: 模式3 (超短查询) + 部分模式1
**风险**: 低
**新增依赖**: 无

### 1.1 增强 GraphSearcher 权重 (search/graph.py)

当前 `_KIND_WEIGHT` 对 call 关系的权重偏低，导致调用链上的代码排名不够靠前。

```
修改: search/graph.py
  _KIND_WEIGHT: call 0.8 → 1.5, inherit 0.7 → 0.9
  _DIR_WEIGHT: backward 1.0 → 1.3 (更重视查找调用者)
```

### 1.2 符号级搜索增强 (search/pipeline.py)

在搜索时自动提取 query 中的代码标识符，作为 FTS 精确匹配的 boost 信号。

```
修改: search/pipeline.py._search_thorough()
  - 用正则从query中提取 snake_case/CamelCase 标识符
  - 对这些标识符做独立的 FTS 精确搜索
  - 将结果加入 fusion_input["symbol_exact"] 通道，高权重
```

### 1.3 查询扩展增强 (search/expansion.py)

现有 QueryExpander 做 two-hop 向量近邻扩展。需要增加：

```
修改: search/expansion.py
  - 添加 CamelCase 分词: "segment2box" → "segment box"
  - 添加 snake_case 分词: "_build_n_nodes_per_face" → "build nodes per face"
  - 添加常见缩写映射: req→request, ctx→context, cfg→config
```

### 1.4 文件级聚合 (search/pipeline.py)

当前返回 chunk 级结果，同一文件的多个 chunk 命中应该聚合提升文件级分数。

```
新增: search/pipeline.py.search_files()
  - 对 search() 结果按文件路径聚合
  - 文件分数 = sum(chunk_scores) 或 max + 0.3*count
  - 返回 FileSearchResult(path, score, top_chunks)
```

### Phase 1 文件清单

| 文件 | 动作 | 修改量 |
|------|------|--------|
| `search/graph.py` | 修改 | ~10行 |
| `search/expansion.py` | 修改 | ~40行 |
| `search/pipeline.py` | 修改 | ~60行 |
| `bench/locbench_eval.py` | 修改 | 用新的search_files |

---

## Phase 2: 实体依赖图

**目标**: File Recall@5 → 70-80%
**解决**: 模式1 (间接依赖)
**风险**: 中高
**新增依赖**: `networkx` (可选)
**前置**: Phase 1

### 2.1 实体ID系统 (core/entity.py) [新建]

```python
# 全局唯一实体标识符
# 格式: "path/to/file.py:ClassName.method_name"
@dataclass(frozen=True)
class EntityId:
    path: str           # 相对文件路径
    qualified_name: str  # 层级限定名 (Class.method)

    @staticmethod
    def from_symbol(file_path: str, symbol: Symbol) -> "EntityId": ...
    def __str__(self) -> str: ...  # "file.py:Class.method"
```

### 2.2 实体依赖图 (core/entity_graph.py) [新建]

```python
class EntityGraph:
    """基于 networkx 的实体依赖图"""

    # 节点: EntityId → {kind, start_line, end_line, ...}
    # 边: EntityId → EntityId, type=imports|invokes|inherits|contains

    def build(self, files_data: list[FileParseResult]) -> None: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...

    # 核心遍历API (模仿LocAgent的explore_tree_structure)
    def traverse(
        self,
        start_entities: list[str],
        direction: str = "downstream",  # upstream/downstream/both
        depth: int = 2,
        edge_types: list[str] | None = None,  # imports/invokes/inherits/contains
    ) -> list[TraversalResult]: ...

    # 搜索集成: 从搜索结果中提取实体，扩展关联实体
    def expand_from_search(
        self, file_paths: list[str], depth: int = 1
    ) -> list[str]: ...
```

### 2.3 索引集成 (indexing/pipeline.py)

```
修改: IndexingPipeline
  - sync() 完成后，收集所有 (path, symbols, references)
  - 构建 EntityGraph 并 save 到索引目录
  - 增量更新: 文件变更时重新解析该文件的 symbols/refs，更新图中相关节点和边
```

### 2.4 搜索集成 (search/pipeline.py)

```
修改: SearchPipeline
  - 构造时接受可选的 EntityGraph
  - search() 完成后，调用 entity_graph.expand_from_search()
  - 将图扩展出的文件加入结果（以较低分数）
```

### 2.5 CLI/MCP 暴露 (bridge.py)

```
修改: bridge.py
  - 新增 cmd_traverse() 子命令
  - codexlens-search traverse --start "file.py:Class" --direction upstream --depth 2
```

### Phase 2 文件清单

| 文件 | 动作 | 修改量 |
|------|------|--------|
| `core/entity.py` | 新建 | ~80行 |
| `core/entity_graph.py` | 新建 | ~300行 |
| `indexing/pipeline.py` | 修改 | ~80行 |
| `search/pipeline.py` | 修改 | ~40行 |
| `bridge.py` | 修改 | ~50行 |
| `pyproject.toml` | 修改 | +networkx |

### 数据模型示例

```
uxarray/grid/coordinates.py:_construct_face_centroids
  ──invokes──► uxarray/grid/connectivity.py:_build_n_nodes_per_face
  ──invokes──► uxarray/grid/coordinates.py:_get_cartesian_face_center

uxarray/grid/grid.py:Grid.face_lon (property)
  ──invokes──► uxarray/grid/coordinates.py:_populate_face_centroids
    ──invokes──► uxarray/grid/coordinates.py:_construct_face_centroids
```

有了这个图，搜索命中 `_construct_face_centroids` 后可以自动扩展到 `_build_n_nodes_per_face`。

---

## Phase 3: LLM Agent 循环

**目标**: File Recall@5 → 90%+
**解决**: 模式2 (症状-根因鸿沟)
**风险**: 中
**新增依赖**: `litellm` (可选)
**前置**: Phase 2

### 3.1 Agent 工具定义 (agent/tools.py) [新建]

```python
# 暴露给 LLM 的 function calling 工具
TOOLS = [
    {
        "name": "search_code",
        "description": "Search codebase for code snippets matching a query",
        "parameters": {"query": str, "top_k": int}
    },
    {
        "name": "traverse_graph",
        "description": "Traverse entity dependency graph to find callers/callees/inheritors",
        "parameters": {"start_entities": list[str], "direction": str, "depth": int, "edge_types": list[str]}
    },
    {
        "name": "get_entity_content",
        "description": "Get full source code of a specific entity",
        "parameters": {"entity_id": str}
    },
    {
        "name": "finish",
        "description": "Submit final localization result",
        "parameters": {"found_files": list[str], "found_entities": list[str]}
    }
]
```

### 3.2 Agent 循环 (agent/loc_agent.py) [新建]

```python
class CodeLocAgent:
    """可选的 LLM Agent，通过迭代搜索+图遍历定位代码"""

    def __init__(
        self,
        search_pipeline: SearchPipeline,
        entity_graph: EntityGraph,
        model: str = "gpt-4o",  # litellm model name
        max_iterations: int = 15,
    ): ...

    def locate(self, problem_statement: str) -> LocResult:
        """LLM Agent 迭代式代码定位"""
        messages = [system_prompt, user_prompt(problem_statement)]
        for _ in range(max_iterations):
            response = litellm.completion(model, messages, tools=TOOLS)
            if response has tool_calls:
                execute tools, append results
            elif response has finish:
                return parse_result
        return timeout_result
```

### 3.3 CLI 集成 (bridge.py)

```
新增: cmd_locate()
  codexlens-search locate --query "problem description" --model gpt-4o
  # 自动使用 agent 循环，输出定位结果
```

### 3.4 LocBench 评估集成 (bench/locbench_eval.py)

```
修改: 添加 --agent 模式
  python bench/locbench_eval.py --agent --model gpt-4o
  # 使用 CodeLocAgent 替代纯搜索
```

### Phase 3 文件清单

| 文件 | 动作 | 修改量 |
|------|------|--------|
| `agent/__init__.py` | 新建 | ~5行 |
| `agent/tools.py` | 新建 | ~60行 |
| `agent/loc_agent.py` | 新建 | ~200行 |
| `bridge.py` | 修改 | ~40行 |
| `bench/locbench_eval.py` | 修改 | ~60行 |
| `pyproject.toml` | 修改 | +litellm[可选] |

---

## 里程碑与预期指标

```
                File Recall@5  Function Recall@10
                ─────────────  ──────────────────
当前 (纯搜索)      33.3%           16.7%
Phase 1 完成       45-50%          25-30%
Phase 2 完成       70-80%          50-60%
Phase 3 完成       90%+            70%+
LocAgent (论文)    92.7%            -
```

## 风险矩阵

| 风险 | 影响 | 概率 | 缓解 |
|------|------|------|------|
| Phase 1 权重调优效果有限 | 低 | 中 | 通过更大样本实验验证 |
| Phase 2 图构建增加索引时间 | 中 | 高 | 懒加载，可选开关 |
| Phase 2 reference提取不够准确 | 高 | 中 | 依赖tree-sitter质量，先只做Python |
| Phase 3 LLM API成本/延迟 | 中 | 高 | 可配置模型，支持本地模型 |
| Phase 3 Prompt质量决定效果 | 高 | 中 | 参考LocAgent的prompt，迭代优化 |

## 依赖关系

```
Phase 1 ──(独立)──► 可独立发布
Phase 2 ──(依赖Phase 1的权重调优)──► 需要entity.py + entity_graph.py
Phase 3 ──(依赖Phase 2的图遍历API)──► 需要litellm + agent模块
```
