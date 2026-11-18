# PLAN: 把 ATOM 的增量 P/D 交接移植到 Mooncake（NVIDIA 路径）

状态：草案（未实现）
分支：`feat/atom-pd-incremental`
范围：Mooncake 仓库内的 vLLM P/D connector（`mooncake-wheel/mooncake/`），不涉及 ROCm/AITER。
参考实现：ROCm/ATOM（`atom/kv_transfer/disaggregation/`）。

---

## 0. 背景

ATOM 的 P/D 交接不是“prefill 全部算完后一次性搬 KV”，而是：

1. **逐 forward 边界发布**：prefill 每完成一个 chunk 就 publish 已完成 block，decode 侧按游标增量接收；
2. **整组保护**：DCP 开启时，一个 destination page 由 `dcp_size` 个 producer block 组成，未凑齐整组不写，避免后到 chunk 覆盖已落页；
3. **first-token handoff**：prefill 侧采样第一个 token，随握手消息传给 decode，decode 不必等 KV 全部到齐就能把首 token 流给客户端；
4. **decode 侧准入 credit**：按 `max_pending_kv_bytes` / `num_reserved_decode_tokens` 限制在途 KV，防止 P 把 D 的 HBM 打爆。

Mooncake 当前的 vLLM connector 是整请求、一次性传输：

| 现状 | 位置 | 影响 |
|---|---|---|
| prefill 结束才触发发送 | `mooncake_connector_v1.py:344` `request_finished()` → `_reqs_need_send` | transfer tail 全部暴露在 TTFT 上 |
| 每请求只传一次 | `mooncake_connector_v1.py:312` | 无法与剩余 prefill 重叠 |
| 不做 layer-wise | `mooncake_connector_v1.py:218-219` | `save_kv_layer` / `wait_for_layer_load` 是空实现 |
| 批量同步写 | `mooncake_connector_v1.py:586` `send_kv_to_decode()` | 一次 `batch_transfer_sync_write` 搬完所有 block |
| 前缀命中只做尾部截断 | `mooncake_connector_v1.py:639-644` | 已有 P/D 两侧 prefix cache 去重，保留 |

**目标**：在 NV 路径上实现“增量交接 + 首 token 提前流式”，把 KV 传输藏进 prefill 计算。

**非目标**（本期不做）：
- AITER/ROCm kernel、MORI-IO、gfx942/gfx950 门控；
- DCP（decode context parallel）的 KV 重排（Phase 3 单列）；
- DCP × 两级 KV offload 的组合（ATOM 自己也标为 unsupported）。

---

## 1. 先测基线（M0，不改代码）

必须先把“transfer tail”量化，否则无法判断增量是否值得。

### 1.1 场景

| 项 | 值 |
|---|---|
| 模型 | 任一 MLA/GQA 均可，先用 dense GQA 排除 DCP 变量 |
| prompt | 32K / 128K，output 128 |
| 拓扑 | 1P + 1D，各 8 卡；P/D 同节点或跨节点各测一次 |
| 传输 | Mooncake TE（RDMA），禁用 TCP |
| 并发 | 1 / 8 / 32 |

### 1.2 打点

在 connector 增加临时 instrumentation（不合并）：

```text
t_queue_admit
t_prefill_first_forward
t_prefill_last_forward
t_transfer_first_block
t_transfer_last_block
t_first_token_emitted
```

派生指标：

- `transfer_tail = t_transfer_last_block - t_prefill_last_forward`
- `overlap_ratio = (t_prefill_last_forward - t_transfer_first_block) / transfer_duration`
- `ttft = t_first_token_emitted - t_queue_admit`

### 1.3 验收

- 得到 `transfer_tail` 占 TTFT 的比例（预期长 prompt 下不可忽略，具体值以实测为准）；
- 确认 P 侧 block 在 `delay_free_blocks` 期间的实际占用时长；
- 记录 32K/128K 与并发 1/8/32 的二维表，作为后续 A/B 的基线。

> 注意：不要把 ATOM 的 MI 系列数字（reload 5.4–7.8×、agentic 2.5× 吞吐）当作 NV 预期，这些是不同路径的收益。

---

## 2. Phase 1：增量发布 + 首 token handoff（核心）

### 2.1 协议设计

新增 side-channel 消息（复用现有 ZMQ 握手通道）：

```text
D -> P: WRITE_REQUEST  { req_id, remote_block_ids, dcp_size, block_size }
P -> D: WRITE_READY    { req_id, num_ready_blocks, event_seq }
P -> D: WRITE_DONE     { req_id, total_blocks }
D -> P: WRITE_ACK      { req_id, total_blocks }   # 所有 reader 完成后才允许 P 释放
```

语义：

- `num_ready_blocks` **单调递增**；除最后一个 block 外只发布整 block；
- D 侧维护自己的 cursor，不需要等 `WRITE_DONE`；
- `WRITE_ACK` 必须等 D 侧所有 TP rank 都落盘（对应 ATOM 的 `expected_readers` / `completed_readers`）；
- P 侧在 `WRITE_REQUEST` 之前不发送，也不在 deadline 之后启动新的 RDMA 写（对应 ATOM 的 write window）；
- 旧 consumer 不认识新消息 → 回退到现有整请求路径，wire 字段可选。

### 2.2 代码改动点

#### Scheduler 侧（`MooncakeConnectorScheduler`）

- `build_connector_meta()`（`:319`）不再只从 `_reqs_need_send` 取整请求；新增 `_reqs_incremental_send`：
  - P 每步 forward 结束后，把“本次新增且完整”的 block 追加进待发队列；
  - 记录 `published_blocks[req_id]`，保证只增不减；
- `request_finished()`（`:344`）仍保留终态处理，但不再承担“首次发送”的职责；
- 失败/取消：`cancel` 后停止新的 publish，但已在途的 RDMA 必须 drain 完（对应 ATOM `ChunkedPrefill.cancel()`）。

#### Worker 侧（`MooncakeConnectorWorker`）

- `send_kv_to_decode()`（`:586`）拆成 `send_blocks_range(req_id, block_start, block_end)`，每次只搬增量；
- `_build_transfer_params()`（`:616`）接受 `(local_start, local_end)` 区间；保留 `group_concurrent_contiguous` 的合并逻辑；
- 生产完成 fence：优先用 TENT 的 `producer_event`（`mooncake-transfer-engine/tent/include/tent/common/types.h:188`，`bc3515b3` 引入并保证 MNNVL 上的拷贝顺序）；当前 Python `TransferEngine`（`mooncake.engine`）绑定尚未暴露该字段，过渡期在 worker 侧持有 CUDA event 并 `event.wait()`。注意 TENT 会拒绝丢失 producer 依赖的 staging / merge / fallback 路径，增量合并成一个 batch 时必须携带该 event；
- `wait_for_layer_load()`（`:218`）仍可保持 no-op；本期做 chunk 粒度，不做 layer 粒度。

#### 生命周期

- 现有 `delay_free_blocks`（`:381`）只等一次发送；改为引用计数：`pending_readers`，每个 D rank 完成 `WRITE_ACK` 后减一，归零才释放；
- 超时：P 侧 `source_safe` 语义——超时只停止新写，不提前释放；
- 幂等：同一 `(req_id, block_range)` 重复发送必须安全（D 侧覆盖写同页，或 P 侧去重）。

### 2.3 正确性不变量（必须有测试）

| 不变量 | 原因 |
|---|---|
| 只发送已完成的 block | forward 异步，未 fence 会读到脏 KV |
| 除尾块外只发整 block | 避免 partial block 被后续 chunk 覆盖 |
| `num_ready_blocks` 单调递增 | 回退会造成重复写与竞态 |
| 不发送 HBM 前缀命中之前的部分 | 与现有 prefix cache 去重一致 |
| 所有 reader ACK 前不释放 P 侧 block | 否则 use-after-free |
| 超时只取消未开始的写 | 在途 RDMA 必须 drain |
| 重复 WRITE 幂等 | 网络重试/多 consumer |

### 2.4 first-token handoff

- P 侧在最后一个 forward 后照常采样；把 `first_token_id`（及可选 draft tokens）放进 `WRITE_DONE` 或独立 `HANDOFF` 消息；
- D 侧先流式返回该 token，再等 KV 到齐后开始第 2 步；
- 前提：P 实例确实执行采样。若当前 P 只算 logits 不采样，需要确认 vLLM P/D 的参数路径；不确定时此项降级为可选。

收益：把“首 token 可见”与“KV 传输完成”解耦。注意 D 的第 2 步仍需要全部 prompt KV，因此本项不改变 decode 启动条件，只缩短客户端感知的 TTFT。

### 2.5 验收

- `transfer_tail` 相对基线下降（目标：长 prompt 下只剩最后一个 chunk 的传输时间）；
- TTFT p50 / p99 不劣化，理想情况下下降；
- 正确性：与“整请求传输”逐 token 对齐（temperature=0，固定 seed）；
- 压力：并发 32、P 端取消、D 端重启、单 block 传输失败注入。

---

## 3. Phase 2：decode 侧准入与背压

参照 `atom/kv_transfer/disaggregation/pd_admission.py`：

| 参数 | 作用 |
|---|---|
| `max_pending_kv_bytes` | D 侧在途 KV 的硬上限 |
| `max_pending_kv_fraction` | 同上，按 HBM 比例 |
| `num_reserved_decode_tokens` | 给 decode 预留的 token 数，避免被 prefill 结果挤爆 |

实现位置：`MooncakeConnectorScheduler.get_num_new_matched_tokens()` / `update_state_after_alloc()`，把 credit 折算成“本轮最多接受多少个外部 token”。超出时让请求在 D 侧排队，而不是拒绝或丢弃。

验收：P 高速灌入时，D 的 decode 步时间不出现尖峰；OOM 归零。

---

## 4. Phase 3（独立评估，不纳入本期）：DCP 重排

仅当需要 **MLA + DCP + P/D 同时开启** 时启动。参考：

- `atom/kv_transfer/disaggregation/sharded_transfer.py`（`DCPShardPlan`、`coalesce_contiguous`、`staged_page_runs`）
- `atom/kv_transfer/disaggregation/landing_scatter.py`（页粒度落 staging + warp-per-segment scatter）
- `atom/kv_transfer/disaggregation/index_staging.py`（DSA index 的 preshuffle gather）

要点：

- token 交错使 source/dest 两端无法同时连续，禁止 token 粒度 RDMA；
- producer 按页落到 consumer staging，再由本地 kernel scatter 进交错 paged KV；
- 代价：额外 staging HBM；scatter 会干扰并发 decode kernel（ATOM 实测优化后干扰约为 `index_copy_` 的 1/3，但非零）；
- 与 Phase 1 的整组保护耦合：`num_ready_blocks` 必须按 `dcp_size` 对齐后再发布。

先决条件：Phase 1 的增量协议稳定，且 D 侧确实需要 DCP。

---

## 5. 决策与代价

| 决策 | 选择 | 代价 |
|---|---|---|
| 增量粒度 | chunk（forward 边界） | 需要 P 侧 scheduler hook；收益随 chunk 数增加 |
| 传输方向 | producer push（沿用现状） | 需要 D 先发 `WRITE_REQUEST`，多一次 RTT |
| layer-wise | 暂不做 | 实现复杂度高，chunk 粒度已能覆盖主要收益 |
| 首 token | 可选 handoff | 依赖 P 侧采样路径 |
| DCP | 本期关闭 | 覆盖不了 MLA 超长上下文场景 |
| 旧协议 | 保留 fallback | 代码分叉，需要 wire 版本字段 |

---

## 6. 风险

1. **vLLM 生命周期不匹配**：当前发送由 `request_finished` 触发；增量需要在每次 forward 后 hook，可能要在 scheduler 之外增加回调或改 proxy 层。
2. **多 consumer fan-out**：一个 P rank 可能服务多个 D rank，`expected_readers` 与引用计数必须精确，否则提前释放。
3. **P 侧 block 复用**：prefix cache 命中会改变 cursor 起点，增量发布的起点必须与 D 的 `local_block_ids` 对齐。
4. **在途写与取消**：超时/取消后仍可能有 RDMA 在写，释放逻辑必须 drain。
5. **同步写阻塞**：现有 `batch_transfer_sync_write` 在 executor 线程；增量后调用次数增加，需要批量合并，避免线程/队列开销反噬。
6. **测量偏差**：P/D 同节点会走 P2P 而非 RDMA，基线必须区分同节点/跨节点。

---

## 7. 里程碑

| 里程碑 | 内容 | 产出 |
|---|---|---|
| M0 | 基线 + instrumentation | `transfer_tail` 二维表、火焰图 |
| M1 | 单请求增量 + first-token handoff | 正确性测试、单请求 TTFT A/B |
| M2 | 并发、失败注入、准入 credit | 稳定性报告、p99 |
| M3 | 评测与结论 | 是否继续 Phase 3 的决策 |

---

## 8. 参考实现索引（ATOM）

| 文件 | 用途 |
|---|---|
| `atom/kv_transfer/disaggregation/chunked_prefill.py` | publish / wait_chunk / handoff / source_safe |
| `atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py` | WRITE_REQUEST/DONE/ACK、增量跳过已缓存前缀、写窗口 deadline |
| `atom/kv_transfer/disaggregation/pd_admission.py` | decode 侧 credit |
| `atom/kv_transfer/disaggregation/types.py` | 完成/失败/保存代际类型 |
| `atom/kv_transfer/disaggregation/pp_kv_aggregator.py` | PP stage 的完成 quorum（本期 P 侧若开 PP 需要） |
| `atom/kv_transfer/disaggregation/sharded_transfer.py` | DCP 重排计划（Phase 3） |
| `atom/kv_transfer/disaggregation/landing_scatter.py` | 落地 scatter（Phase 3） |

---

## 9. 待确认

- [ ] vLLM P 实例是否执行采样（决定 first-token handoff 是否可行）
- [ ] Python `TransferEngine` / `mooncake.engine` 是否会暴露 TENT 的 `producer_event`（决定是否要在 connector 里自建 event 等待）
- [ ] 目标 P/D 拓扑是否允许在 forward 后插入 side-channel 消息（是否已有 hook 点）
- [ ] 是否需要同时支持 SGLang connector（`mooncake-wheel/` 下是否已有对应实现）
- [ ] 基线使用哪一代 GPU 与 NIC（决定 staging 预算与并发上限）
