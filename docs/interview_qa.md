# 设计决策与常见问题（Design Decisions & FAQ）

围绕本项目真实实现，记录关键设计取舍与常见疑问，覆盖架构选型、编排、MCP、评估、护栏、可观测性与部署。

---

## 一、架构与选型

**Q1：为什么用 LangGraph 而不是简单的 ReAct / 手写循环？**

ReAct 是「模型决定下一步」的单循环，无法表达**确定性的控制流**（审批暂停、条件分支、多角色协同、状态持久化）。LangGraph 把流程建模成图：节点是动作、边是转移，天然支持 `interrupt` 中断、`checkpointer` 持久化、`store` 长期记忆。本项目的退款审批必须在执行前暂停等待人工决策——这是 ReAct 做不到的。

**Q2：为什么要把业务逻辑和框架解耦？**

`business.py` / `order_store.py` / `policy.py` 是纯 Python，不 import langchain / langgraph。好处有三：① 同一批业务函数能被 V1/V2 进程内工具、V3/V4 的 MCP Server 复用；② 可脱离 LLM 环境直接单测（`scripts/verify_tools_logic.py`）；③ 换框架不改业务代码。

**Q3：V1 到 V4 的演进逻辑是什么？**

V1 单 Agent + ReAct 打通工具调用；V2 升级为多智能体（Supervisor + 三专员 + 审批 + 长期记忆）；V3 把工具从进程内迁到 MCP Server 并全链路异步化；V4 补齐工程化（评估 / 护栏 / 可观测 / 部署 / CI）。每一步都在解决上一版的真实痛点，而不是堆功能。

---

## 二、LangGraph 编排

**Q4：Supervisor 路由为什么用 `Command(goto=...)` 而不是 conditional edges？**

`Command` 让一个节点在**一次返回里同时完成「更新状态」和「指定下一跳」**，路由逻辑和状态更新写在一起，可读性更好；conditional edges 需要额外写一个只返回字符串的路由函数，逻辑被拆成两处。

**Q5：怎么防止 Supervisor 和专员之间反复横跳死循环？**

状态里维护 `turns` 计数，Supervisor 每被调用一次 +1，超过 `MAX_SUPERVISOR_TURNS`（8）直接 `goto=END`。这是图编排里必须做的**收敛保护**。

**Q6：工具权限隔离怎么做的？为什么说「结构上」拿不到？**

售前专员 `bind_tools(presale_tools)` 只绑定了商品类工具，其工具节点 `ToolNode(presale_tools)` 里**根本没有** `apply_refund`。所以即使 LLM「幻觉」出退款意图，也没有可调用的工具——权限隔离在**图的拓扑结构**上成立，而不是靠提示词约束。

**Q7：checkpointer 和 store 有什么区别？**

`checkpointer`（MemorySaver）存的是**单条会话的状态快照**，支撑多轮对话与中断恢复；`store`（InMemoryStore）存的是**跨会话的长期记忆**，本项目存用户画像和最近工单。前者按 `thread_id` 隔离，后者按 `(namespace, key)` 组织。

---

## 三、MCP 服务化

**Q8：为什么要拆成两个 MCP Server？**

按**业务边界**拆分：检索服务（商品知识 / 商品详情 / 计算）与订单服务（订单 / 物流 / 退款）。好处：① 权限隔离天然对齐（售前只用检索服务）；② 独立部署、独立扩缩容；③ 故障隔离——订单服务挂了不影响商品咨询。

**Q9：MCP 工具是动态发现的，怎么保证权限不越权？**

用**角色名白名单**：`PRESALE_TOOL_NAMES` / `AFTERSALE_TOOL_NAMES`。发现是动态的（server 加了新工具 client 能发现），但**授权是显式的**（白名单外的工具不授予任何角色，仅告警）。把「发现」和「授权」解耦，避免 server 端多加一个工具就让 agent 顺手拿到新权限。

**Q10：MCP Server 用 stdio 传输时有什么坑？**

**stdout 是协议流，绝对不能有裸 `print`**。本项目所有 MCP Server 的日志强制走 stderr（`LOG_CONSOLE_STREAM=stderr`），RAG 检索链里的 print 全部收进 `__main__` 演示块。一旦有 print 污染 stdout，client 解析就会失败。

**Q11：MCP 工具为什么必须在 `async with` 作用域内使用？**

MCP 工具与**会话**绑定——`client.session()` 打开的连接关闭后工具即失效。所以 `MCPToolHub` 用 `AsyncExitStack` 在 `__aenter__` 打开两个 server 会话、`__aexit__` 统一关闭，工具在作用域内暴露。这也解释了为什么不能用「一次性函数返回工具列表」。

---

## 四、人工审批与状态

**Q12：人工审批的中断/恢复怎么实现的？**

`approval` 节点里调用 `interrupt({...})` 暂停图，把决策问题（订单号、金额、选项）抛给上层；上层拿到中断载荷后，用 `Command(resume="approve"/"reject")` 恢复。批准 → `goto="aftersale_tools"` 执行退款；驳回 → 回 `after_sales` 并注入一条 ToolMessage 告知驳回。

**Q13：为什么审批逻辑写在节点里，而不是写在 `apply_refund` 工具里？**

职责分离：`apply_refund` 保持**纯执行**（只负责退款），可独立单测；审批是**编排层策略**，写在图节点里。阈值集中在 `policy.py`，调整风控口径不需要改图和工具。

---

## 五、评估体系

**Q14：为什么 Agent 层要用「脚本化 LLM」而不是真实 LLM？**

真实 LLM 有决策随机性，会把「图编排是否正确」和「模型发挥是否稳定」两件事混在一起。注入 `ScriptedLLM`（按剧本返回固定 AIMessage）后，本层变成**确定性门禁**，可稳定复现、可进 CI。模型能力的评估交给 RAG 层和 judge 层。

**Q15：评估为什么分层？**

不同层的**依赖和稳定性**不同：Agent / 护栏层确定性、零外部依赖 → 进 CI 硬门禁；RAG 层依赖 Chroma、judge 层依赖真实 LLM → 离线补充。这样 CI 不会因为「外部依赖抖动」而误报失败。

**Q16：外部依赖不可用时怎么处理？**

**显式降级而非静默通过**。RAG 层在 Chroma 不可用时抛 `RetrievalUnavailable`，由 `run_eval` 标记该层 `skipped` 并在报告里注明原因——绝不能让它「假装通过」。

**Q17：具体有哪些指标？**

检索：Hit@1/3/5、MRR；分类/判定：accuracy、detection_rate（召回）、false_positive_rate（误拦）。指标都是**纯函数**（`eval/metrics.py`），与框架无关，可单测复用。

---

## 六、安全护栏

**Q18：输入护栏为什么用规则而不是分类模型？**

规则**零依赖、零延迟、可解释**，适合作为第一道廉价闸门——拦掉最常见的注入/越狱/越权诱导。真正的对抗场景应叠加分类器或 LLM 复核。规则命中即拦截并给出 category，便于在评估集里统计各类攻击召回率。

**Q19：工具入参校验为什么要在「执行前」？**

LLM 生成的工具参数是**不可信输入**。`order_store` 虽然也做了兜底校验，但那是在**执行时刻**；`tool_guard` 用 Pydantic 在**调用前**声明式校验，把错误挡在副作用（如误退款）发生之前。两道防线互补。

**Q20：护栏怎么做到不侵入业务代码？**

入参校验是横切关注点，`guardrails/wrap.py` 用 `StructuredTool` 把工具**重新包一层**（先校验再委托），对任意图生效，不用改 V2/V3 图源码。

**Q21：包装 MCP 工具时踩过什么坑？**

MCP 适配器工具带 `response_format='content_and_artifact'` 且自行处理返回值。如果把这个字段继承到标准 `StructuredTool` 上，会触发二段校验，而工具返回的内容列表不满足「(content, artifact) 二元组」约定而报错。解决：包装层**不继承** `response_format`，用默认 content 格式透传结果。

---

## 七、可观测性

**Q22：可观测性怎么做的？为什么默认零外部依赖？**

用 LangChain 的 `BaseCallbackHandler` 实现 `TraceRecorder`，采集节点延迟、LLM Token/成本、工具调用次数，落本地 JSONL。默认**不装 Langfuse 也能跑**，便于 CI 与离线分析；配置 `LANGFUSE_*` 且安装 `langfuse` 后自动附加 handler，实现 trace 可视化——**无需改动业务代码**。

**Q23：成本是怎么算的？**

Token 数 × 单价估算，单价可用环境变量覆盖。刻意把它标注为「估算」而非「账单」，避免把近似值当精确值。

---

## 八、部署

**Q24：最小部署包含什么？**

FastAPI 服务层：`/chat`（一次问答）、`/chat/stream`（SSE 流式，逐节点推送）、`/resume`（审批恢复）、`/ok`（健康检查），加一个自包含的演示页。Dockerfile 用 `HEALTHCHECK` 探 `/ok`，密钥通过 `--env-file` 注入、不打进镜像。

**Q25：容器里检索为什么需要挂载模型缓存？**

检索链用本地 BGE 模型（`~/.cache/huggingface/local`）。为规避 HuggingFace 联网超时，全链路设 `HF_HUB_OFFLINE=1`，因此容器需把宿主模型缓存挂载到 `/root/.cache/huggingface` 离线加载；仅订单类流程不依赖向量库。

---

## 九、难点与踩坑总结

1. **HuggingFace 模型加载超时**：`WinError 10060` 触发 5 次重试阻塞 → 改用本地缓存 + `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，且必须在导入检索链**之前**设置。
2. **目录联接（junction）失效**：snapshot 里的相对符号链接导致模型路径解析失败 → 改用 `robocopy` 复制真实文件。
3. **stdio 协议流被污染**：MCP Server 里的 print 让 client 解析失败 → 日志一律走 stderr。
4. **Chroma 被锁**：受限环境下无法创建 client → 评估层显式降级为 `skipped`，不阻断 CI。
5. **包装 MCP 工具报错**：`response_format` 继承导致二段校验失败 → 包装层不继承该字段。
6. **AI 生成代码需人工验证**：LangGraph / MCP 的实现细节（异步、会话生命周期、中断语义）必须手动验证正确性，不能盲信。
