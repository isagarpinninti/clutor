# @clutor/core

The engine behind [Clutor](https://github.com/isagarpinninti/clutor): no runtime dependencies.

- **GGUF reader** (`parseGguf`, `buildModelLayout`): header, metadata, tensor table, per-expert byte ranges. Refuses damaged or hostile files with `GgufError`.
- **Memory planner** (`planMemory`, `CATALOG`, `approximateLayout`): where every byte goes and the disk-bound speed ceiling.
- **Cache policies and simulator** (`LruCache`, `WTinyLfuCache`, `AdaptiveTinyLfuCache`, `simulateStreaming`, `generateRoutingTrace`).
- **Coding agent** (`runAgent`, `ShadowWorkspace`, `planCommand`, `computeUnifiedDiff`, receipts).

```ts
import { CATALOG, QUANTS, GiB, approximateLayout, planMemory } from "@clutor/core";

const layout = approximateLayout(CATALOG.find((c) => c.id === "deepseek-v3")!, QUANTS.Q4_K_M!);
const plan = planMemory(layout, { hostRamBytes: 16 * GiB });
console.log(plan.strategy, plan.speed.map((s) => [s.hitRate, s.maxTokensPerSec]));
```

The main entry runs in browsers. `@clutor/core/node` adds the file system, scratch-copy command runner, hardware probing and model-file reads. It is **not** an inference engine: see the [limits](https://github.com/isagarpinninti/clutor/blob/main/docs/LIMITS.md).
