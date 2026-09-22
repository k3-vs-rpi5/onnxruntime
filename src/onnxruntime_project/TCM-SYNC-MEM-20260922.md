# G7 answered: what `/dev/tcm_sync_mem` is, and why the board does not need it

Item G7 of `NEXT-ROUND-ANALYSIS-20260922.md` was left open as "a board-provisioning question with a
number attached": every provider and llama.cpp run logs
`alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2`, the node does not exist on this board, and the
non-scaling part of a YOLO frame (~4.2 ms) had barriers as one candidate. The operator asked for the
question to be researched against SpacemiT's own material. It is now **answered, and closed**: the
node is a different, optional resource, and the vendor's own position is that it does not matter.

## 1. What references it, measured on the board

| artefact | reference |
|---|---|
| `/usr/lib/libspine_tcm.so.3.0.0` (deb `spacemit-tcm` 3.0.0, built 2026-05-20) | the whole sync-mem resource: `tcm_sync_resource_sync_mem.c`, `tcm_sync_mem_init_or_reset`, `tcm_sync_mem_page_size`, `tcm_sync_mem_usable_block_size`, `tcm_sync_mem_mode_t` / `tcm_sync_mem_mode_name`, and the messages `sync region too small: need %zu bytes, have %zu bytes`, `sync metadata too large`, `flock sync`, `mmap tcm reset` |
| `/usr/lib/libggml-cpu.so.0.16.0` (llama.cpp's SpacemiT backend) | one string, next to its `spine_mem_pool_manager` / `spine_mem_pool_posix` symbols: the pool tries this node and falls back to a POSIX-backed pool |
| the ONNX provider's own logs | the same fallback line, once per session |

So it is an **mmap-able, flock-synchronised shared region that the Spine TCM runtime would prefer for
its sync/cache pool** - a *second* TCM resource, not the compute one.

## 2. What the kernel side says

* the board's running BSP is `spacemit-com/linux-6.18`, branch `k3-br-v1.0.y`; our pinned checkout is
  `4158237f3` (`k3-br-v1.0.7`), and the branch head is `0bef28bf`;
* in that tree the only TCM driver is `drivers/misc/tcm.c` behind `CONFIG_SPACEMIT_TCM` ("Generic tcm
  alloc management driver", `depends on SOC_SPACEMIT && SPACEMIT_HMP`, default y) - it registers the
  misc device that becomes **`/dev/tcm`**, which is present and is the 8 x 384 KiB block device our
  TCM evidence has been reading all along;
* `grep -c sync_mem drivers/misc/tcm.c` on the **branch head** returns **0**, there is no
  `drivers/misc/tcm_sync_mem.c` (404), and the org's other kernel trees (`linux`, `linux-6.6`,
  `linux-k1x`), `spine-runtime`, `ai-support` and the docs repositories contain no reference to the
  name either. The node is therefore not part of the K3 BSP kernel at all - not something our board
  is missing by configuration.

## 3. The vendor's own answer

The question has already been asked in SpacemiT's forum by another K3 user, with the same symptom
(no `spacemit-tcm.ko`, no `/dev/tcm_sync_mem`, llama.cpp falling back to heap, "functionally fine but
without that optimisation"), on 2026-08-03. SpacemiT's reply, 2026-08-04
(`forum.spacemit.com/t/topic/1555`, thread "TCM支持 - K3"):

> 忽略这个问题 … sync_mem，是另一个 tcm，用于数据缓存的 tcm 一定会打开使用，否则会直接报错，
> 这个 sync_mem 的 tcm 不太影响性能

translated for the record: *ignore this; `sync_mem` is another TCM, used for data caching - the TCM
used for compute is always opened and used, otherwise it would fail outright; this `sync_mem` TCM
does not affect performance much.*

That matches our own observations exactly: the compute TCM is open and busy (8/8 blocks, named pids,
one block per compute thread), and the only trace of the sync-mem resource is one fallback line in
the log.

## 4. Decision

* **G7 is closed.** No board-provisioning change is proposed, and the ~4.2 ms non-scaling part of a
  YOLO frame is **not** attributed to a missing sync-mem device - the vendor says the resource is
  optional and low-impact, and the kernel does not implement it for K3.
* What would reopen it: a vendor statement that names a measurable prize, or a kernel/BSP revision
  that ships the node. Until then, the line in the logs is a *documented fallback*, quoted as such,
  not an open optimisation.
* The forum answer is cited rather than paraphrased because it is the authority here, and it is the
  first SpacemiT statement this project has found that closes one of its own open questions.
