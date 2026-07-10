# POMO 复现任务交接给服务器 Codex

你现在接手一个严谨的 POMO 论文复现任务。请先完整阅读本交接，再检查服务器上的实际仓库和环境。不要立即训练，不要覆盖学长已有文件，不要把尚未验证的推断写成事实。

## 1. 当前任务边界

当前任务只研究 POMO 论文及官方代码中的 TSP 和 CVRP，不再接之前的 HeteroMRTA、RALTestSet、makespan、waiting time 或 AWAR。

我们的目标分成两层：

1. 先用官方 POMO checkpoint 跑通并理解 TSP20/TSP50/TSP100/CVRP100 的 x8 augmentation inference，复现论文表格最后一行。
2. 建立一套固定、可复用、可审计的测试集和 evaluator，以后新 idea 的模型必须在完全相同的 instances 和相同统计规则下与 POMO 比较。

学长已经确认当前范围和优先级：

- 只复现表格最后一行 `POMO, x8 augment.`，前面的外部 baseline 不做。
- CVRP20/CVRP50 可以自行训练，但必须与原文训练环境严格一致，而且优先级靠后，先完成其他复现准备。
- 可以保存 TSP 固定测试集，供 POMO 和以后新 idea 使用同一套 instances。
- 可以补 CVRP 测试集生成/保存脚本，把官方 generator 与官方 fixed-data loader 衔接起来。
- TSP50 可以复用 `TSPTester`，但需要继续核实完整配置，不能认为只改 `problem_size` 和 `pomo_size` 就已经严谨对齐。
- VSCode 服务器连接已经解决，不属于后续复现任务。

操作结论：当前不要为复现论文前面的 baseline 耗费时间，也不要让严格 Gap 阻塞 x8 score 的复现。若以后确实要求自己计算 Gap，再为同一个固定测试集单独生成 reference costs；不能用不相同的测试集或论文四舍五入均值代替。

## 2. 严谨性规则

- 所有结论区分为：官方论文明确写明、官方代码直接确认、我们新增代码得到、仍需从证据核实。
- 不要把我们按官方生成分布新建的固定测试集称为“论文作者原始测试集”。原始 instances 未公开时，只能称为“按官方 generator 和固定 seed 生成的复现实验测试集”。
- 不要只比较两个模型各自在不同随机测试集上的平均值。正式对比必须使用同一个保存下来的测试文件。
- 训练集与测试集严格隔离。POMO 训练时在线随机生成 batch；固定测试集只能用于测试，不能喂给训练。
- 不要用论文表中四舍五入后的 Len. 反推并宣称得到了严格 Gap。严格 Gap 需要同一 instance 的 reference cost。
- 不要把 10 个 instance 的 smoke test 结果当作论文复现结果。
- 不要无说明地修改官方源码。优先把新增脚本放在仓库外或仓库内单独的 `reproduction_tools/` 目录，并保留 `git status` 证据。
- 每次正式运行记录：git commit、checkpoint、测试集路径和 SHA256、Python/PyTorch/CUDA/GPU、episodes、batch size、augmentation、随机种子、score、runtime。

## 3. 已核实的官方仓库信息

本机克隆的官方仓库是 `https://github.com/yd-kwon/POMO.git`，当时 HEAD 为：

```text
d7c3d6ea580499a53e874fe9e065f69e799a8551
```

服务器上的仓库是学长克隆的，commit 尚不确定。接手后第一件事是执行：

```bash
cd /服务器上的/POMO
git rev-parse HEAD
git status --short
git remote -v
```

不要为了对齐 commit 直接 reset 或覆盖学长的修改。先记录差异并汇报。

官方 README 对版本的说明很重要：

- `OLD_ipynb_ver` 是 2020 年论文实验使用的原始 notebook 代码。
- `NEW_py_ver` 是 2021 年重新组织的 Python 版本，适合服务器运行。
- 官方称两版应给出大致相同结果，但 NEW 版确有模型结构/归一化变化，例如 decoder query 不再含 graph encoding，BatchNorm 改为 InstanceNorm。

因此，我们当前首先复现 `NEW_py_ver` 官方 checkpoint 的结果；若声称“逐项严格复现论文原始 Table 2/3”，必须注明 NEW 版并非完全等同 OLD 原始实现，必要时再审计 OLD 版。

官方 NEW 版 README 给出的旧环境是 Python 3.7.6、PyTorch 1.7.0。本机 smoke test 使用 Python 3.8.20、PyTorch 2.4.1+cpu 也能运行，但正式记录必须写明实际版本，尤其不能把现代 GPU 上的 runtime 与论文 Titan RTX runtime 直接等同。

## 4. 已核实的模型与缺失材料

### TSP

官方 `NEW_py_ver/TSP/POMO` 中存在：

- TSP20 checkpoint: `result/saved_tsp20_model/checkpoint-510.pt`
- TSP50 checkpoint: `result/saved_tsp50_model/checkpoint-1000.pt`
- TSP100 checkpoint: `result/saved_tsp100_model2_longTrain/checkpoint-3100.pt`
- 官方入口有 `test_n20.py` 和 `test_n100.py`
- 官方没有单独的 `test_n50.py`，但这不影响加载 TSP50 checkpoint。使用同一个 `TSPTester`，配置 `problem_size=50`、`pomo_size=50`、checkpoint path 和 epoch 1000 即可。

对本地三个 TSP checkpoint 做过结构核对：TSP20/TSP50/TSP100 的 `model_state_dict` 都有 86 个参数项，键名和每个 tensor shape 完全一致，因此 TSP50 可沿用相同 `TSPModel` 架构加载 checkpoint。但严谨的 TSP50 测试入口至少还要对齐：

- `problem_size=50`
- `pomo_size=50`
- `eval_type='argmax'`
- `augmentation_enable=True`
- `aug_factor=8`
- checkpoint path 为 `saved_tsp50_model`
- checkpoint epoch 为 `1000`
- 测试规模是论文口径 10,000，还是 NEW 版 TSP 脚本口径 100,000
- 使用保存的 TSP50 fixed test set，而不是每次重新随机生成
- `test_batch_size`/`aug_batch_size`、GPU 显存和 runtime 记录
- logger/output 名称，避免覆盖其他规模结果

官方仓库文本和 TSP50 checkpoint 中没有保存现成 `test_n50.py` 或测试 batch 配置。我们此前 runner 采用 `aug_batch_size=400` 只是介于 N=20 的 1000 与 N=100 的 100 之间的工程选择，不是已证实的官方参数。batch size 在确定性 inference 下原则上不应改变 solution score，但会改变显存占用和 runtime，所以必须明确记录，不能拿未经对齐的时间复现论文 16 秒。

### CVRP

官方 `NEW_py_ver/CVRP/POMO` 中存在：

- CVRP100 checkpoint: `result/saved_CVRP100_model/checkpoint-30500.pt`
- CVRP100 固定测试集: `NEW_py_ver/CVRP/vrp100_test_seed1234.pt`
- 官方入口有 `train_n100.py` 和 `test_n100.py`

官方公开仓库当前没有：

- CVRP20 checkpoint
- CVRP50 checkpoint
- CVRP20/CVRP50 固定测试集
- CVRP20/CVRP50 的 NEW 版 train/test 入口脚本
- 独立的 CVRP 测试集生成并保存脚本

不要因为 `CVRProblemDef.py` 支持 N=20/50 就声称已有可直接复现的官方模型。生成问题的能力和训练好的 checkpoint 是两回事。

## 5. 官方问题设置

### TSP

`NEW_py_ver/TSP/TSProblemDef.py::get_random_problems` 直接调用：

```python
torch.rand(size=(batch_size, problem_size, 2))
```

即 N 个二维坐标独立均匀采样于 `[0,1]^2`。TSP 中 `pomo_size=N`，每个节点作为一条 trajectory 的指定起点。最终 cost 是闭合 tour 的欧氏长度，越小越好。

### CVRP

`NEW_py_ver/CVRP/CVRProblemDef.py::get_random_problems` 的逻辑是：

```python
depot_xy = torch.rand(batch_size, 1, 2)
node_xy = torch.rand(batch_size, problem_size, 2)
node_demand = torch.randint(1, 10, ...) / demand_scaler
```

其中 demand scaler：

- CVRP20: 30
- CVRP50: 40
- CVRP100: 50

容量归一化为 1。CVRP 的 depot 是索引 0；路线可以多次返回 depot 补满容量。`CVRPEnv._get_travel_distance()` 对完整选择序列做闭环距离求和，所以包含从 depot 出发以及最后返回 depot 的距离。

## 6. POMO 推理模式与当前交付范围

论文中的三行不是三个不同 checkpoint，而是同一个 POMO-trained network 的三种推理方式：

1. `POMO, single trajec.`：随机选择一个起点，只做一条 argmax/greedy trajectory。
2. `POMO, no augment.`：不做坐标增强，对 N 个 POMO 起点产生的 N 条 greedy trajectories 取最优。
3. `POMO, x8 augment.`：做 8 种保距坐标变换，对 `8 x N` 条 trajectories 取最优。

学长已明确当前只要求复现最后一行 `POMO, x8 augment.`，因此正式主结果只需把 x8 score 跑严谨。single trajectory 和 no augmentation 可保留为内部 sanity check，但不属于当前必须交付的论文行。

如果 evaluator 仍输出 single trajectory，必须注明当前实现是否忠实。论文明确说 single trajectory 使用随机起点，因此不能把“固定取第 0 条 trajectory”无说明地当作完全忠实实现。可以给 single 模式设置固定随机种子，并为每个 instance 随机选一个 POMO 起点；否则就从正式结果中移除该列。

`NO-AUG SCORE` 与 `AUGMENTATION SCORE` 都是平均路线长度，不是 reward 符号值。环境 reward 是负距离，tester 最终取负号转回正的 cost。

## 7. 论文与 NEW 版测试规模的区别

这里不能混为一谈：

- POMO 论文正文写的是对每个问题求解 10,000 个随机测试 instances，并报告 Table 2/3 时间。
- 论文还说明 TSP 表中的部分平均 Len. 为了与更大量采样得到的 3.83/5.69 等总体均值一致，按已报告 Gap 做过轻微调整，所以表中四舍五入 Len. 本身不是逐实例复算 Gap 的充分数据。
- NEW 版 `TSP/POMO/test_n20.py` 与 `test_n100.py` 当前设置 `test_episodes=100000`，现场随机生成，不保存测试集。
- NEW 版 `CVRP/POMO/test_n100.py` 设置 `test_episodes=10000`，读取官方固定 `vrp100_test_seed1234.pt`。

因此正式报告要写清是在做：

- “运行 NEW_py_ver 官方测试配置”，还是
- “按论文 10,000 instances 的表格规模复现”。

建议生成 100,000 个 TSP 固定 master instances，并约定前 10,000 个作为 paper-scale 子集。这样可同时支持 10k 和 100k 评测，但这仍是我们新生成的同分布测试集，不是论文作者当年的原始 instances。

## 8. 论文表格中的 POMO 目标值

这些是论文表中用于 sanity comparison 的四舍五入结果，不应被当作逐实例 ground truth。当前必须复现的是最右侧意义上的 `x8 augment.` 行；另外两列保留在这里仅用于理解和内部检查。

| Problem | single trajec. | no augment. | x8 augment. |
| --- | ---: | ---: | ---: |
| TSP20 | 3.83 | 3.83 | 3.83 |
| TSP50 | 5.73 | 5.70 | 5.69 |
| TSP100 | 7.84 | 7.80 | 7.77 |
| CVRP20 | 6.35 | 6.17 | 6.14 |
| CVRP50 | 10.74 | 10.49 | 10.42 |
| CVRP100 | 16.15 | 15.83 | 15.73 |

论文相应 POMO Gap 为：

| Problem | single trajec. | no augment. | x8 augment. |
| --- | ---: | ---: | ---: |
| TSP20 | 0.12% | 0.04% | 0.00% |
| TSP50 | 0.64% | 0.21% | 0.03% |
| TSP100 | 1.07% | 0.46% | 0.14% |
| CVRP20 | 3.72% | 0.82% | 0.21% |
| CVRP50 | 3.52% | 1.14% | 0.45% |
| CVRP100 | 3.00% | 0.98% | 0.32% |

## 9. baseline 与 Gap 的边界

只靠 POMO 官方仓库，可以直接运行 POMO 模型，但不能自动复现 Table 2/3 的所有外部 baseline 或严格 Gap。

论文表格前面的 Concorde、LKH3、Gurobi、OR-Tools、Farthest Insertion、GCN、AM、Improvement methods、NeuRewriter、NLNS、L2I 等不是 POMO 仓库内的一整套可运行工具。部分有外部 solver 或原作者仓库，但需要单独获取、对齐版本、参数和测试集。

严格 Gap 至少需要：

- 与 POMO 完全相同的每个测试 instance
- 对该 instance 的 reference cost
- 论文一致的 Gap 聚合公式

TSP 通常需要 Concorde/等价 exact reference；CVRP 表中使用 LKH3 reference，但 LKH3 并非数学意义上的全局最优证明。未核实公式前，不要自行决定使用“平均 cost 的比值”还是“逐实例 relative gap 的平均”。

学长已确认只复现最后一行 `POMO, x8 augment.`，前面的 baseline 不管。因此当前先交 x8 Len./score 和本机 runtime。严格 Gap 没有同 instance reference 时标记为未计算，不要为了得到 Gap 去复现整张 baseline 表，也不要伪造。

## 10. 本地已经新增的文件

必须从本机复制到服务器的三个工作脚本：

1. `work/run_pomo_reproduction.py`
   - 统一调用官方 checkpoint。
   - 当前支持 TSP20/TSP50/TSP100/CVRP100。
   - TSP 在运行时随机生成；CVRP100 读取官方固定测试集。
   - 当前只输出 no-aug 和 x8 aug，不输出严格 single trajectory。
2. `work/generate_pomo_tsp_testset.py`
   - 调用官方 `TSProblemDef.get_random_problems()`，按固定 seed 生成 TSP 测试集并保存 metadata dict。
   - 不修改官方 POMO 仓库。
3. `work/run_pomo_tsp_fixed_eval.py`
   - 在保存的 TSP 测试集上加载官方 checkpoint。
   - 输出 single/no-aug/x8 三项。
   - 已知限制：当前 single 直接取 POMO index 0，不是论文所说的随机起点。正式使用前必须修正。

传输后可用 SHA256 校验原文件：

```text
run_pomo_reproduction.py      6D6AD5CCC3986AE664FA3C120277FA517349F1D136E07B75F5E1FA020084868B
generate_pomo_tsp_testset.py  7C9D7562426B367D7390853AF0ACCFBD232CDA86B23AB2186C9FF7155AEC4722
run_pomo_tsp_fixed_eval.py    83470890D5771D798FAC4B1F1AA79C22E4785DB442596F382324D85B1FB13C31
```

建议一并复制作为历史证据：

- `outputs/pomo_reproduction_smoke.csv`
- `outputs/pomo_tsp20_fixed_tiny.csv`
- `outputs/fixed_testsets/tsp20_test_seed1234_tiny.pt`，仅 10 个 instance，只用于检查 loader，不用于正式评测
- 本交接文件 `SERVER_CODEX_HANDOFF.md`

不要复制或不要让它们影响当前任务：

- `work/inspect_heteromrta_testset.py`
- `work/run_heteromrta_one.py`
- `work/heteromrta_inspect/`
- `outputs/pomo_eval_next_steps.md`，它是任务尚未澄清时写的 MRTA 旧计划
- `work/pdfs/MRTA_RL.pdf` 及 MRTA 文本

## 11. 本地已经跑过的结果

使用官方 checkpoint、CPU、Python 3.8.20、PyTorch 2.4.1+cpu，对每种问题只跑了 10 个 instances。它们只证明代码流程和 checkpoint 能加载：

| case | no_aug score | x8 aug score |
| --- | ---: | ---: |
| TSP20 | 3.8288486 | 3.8288486 |
| TSP50 | 5.6583948 | 5.6494765 |
| TSP100 | 7.6563826 | 7.6312461 |
| CVRP100 | 15.6327419 | 15.5858126 |

注意：TSP 这次是现场随机生成并用 seed 123；CVRP100 使用官方 fixed test set 的前 10 个。样本太少，结果看起来比论文好或坏都没有统计意义。

另外生成过一个 10-instance 的固定 TSP20 tiny test set，seed=1234。当前 fixed evaluator 输出：

```text
single(index 0, not yet random)=3.9453475
no_aug=3.9430103
x8_aug=3.9426086
```

这也只是 loader/evaluator 的 smoke test。

## 12. 现有脚本迁移到 Linux 时的注意事项

三个脚本的默认 `DEFAULT_POMO_ROOT` 是本机 Windows 路径，不能在服务器直接使用。第一轮可以不改文件，始终显式传参：

```bash
python reproduction_tools/run_pomo_reproduction.py \
  --pomo-root /服务器上的/POMO \
  --mode smoke --device cuda --seed 123 \
  --output outputs/pomo_reproduction_smoke_server.csv
```

之后可以把默认路径改成相对于脚本的路径或必填 CLI 参数。不要把本机盘符写进正式服务器脚本。

现有 runner 的计时使用 `time.perf_counter()`，但 CUDA 是异步执行。正式测 GPU runtime 时必须在计时起止点调用 `torch.cuda.synchronize()`，并考虑单独 warm-up；否则时间可能偏小。只比较 solution cost 时不受此问题影响。

如果为了避免 OOM 调低 batch size，必须在结果表中记录实际 batch size。不同 batch size、GPU 和软件版本的 runtime 不应与论文时间直接比较。

## 13. 服务器接手后的执行 TODO

### Phase A: 审计服务器仓库和环境

- 记录 POMO 根路径、commit、remote、`git status --short`。
- 列出 TSP/CVRP checkpoint 和测试集，确认与本交接一致。
- 运行 `nvidia-smi`。
- 记录 `python --version`、`torch.__version__`、`torch.version.cuda`、`torch.cuda.is_available()` 和 GPU 名称。
- 找到当前可用 conda 环境；不要为了追求 PyTorch 1.7 立即破坏已有环境。
- 在小样例上确认官方代码可在现有 PyTorch 版本运行。

### Phase B: 接收并审计三个脚本

- 放入独立 `reproduction_tools/`，不要覆盖官方 `NEW_py_ver`。
- 用 `sha256sum` 校验传输内容。
- 修改或参数化 POMO 根路径，使其适配 Linux。
- 当前正式输出聚焦 x8。对 TSP fixed evaluator 现有的 single trajectory 列，要么先从正式结果移除，要么再按固定 seed 为每个 instance 随机选择起点后保留；不能继续把固定 index 0 标成论文 single trajectory。
- 给 fixed evaluator 增加输入 shape、N、instance count、NaN/Inf、checkpoint 存在性检查。
- 给结果写入 git commit、测试集 SHA256、seed、环境版本和 GPU。
- 修正 CUDA runtime 计时同步。

### Phase C: 先跑 smoke test

- TSP20/TSP50/TSP100/CVRP100 各跑 10 个 instances。
- 检查 `single >= no_aug >= x8_aug` 是否在聚合意义上成立。逐 batch/有限样本可能有极小数值差异，但 no-aug 与 x8 的候选包含关系实现必须正确。
- 检查 checkpoint 加载无 missing/unexpected keys。
- 检查 CVRP100 确实读取官方 `vrp100_test_seed1234.pt`，没有退回随机生成。
- 不要在 smoke test 通过前启动 full run。

### Phase D: 建立固定测试集

- TSP20/TSP50/TSP100 使用官方 generator、CPU、固定 seed 1234，各生成 100,000 个 master instances。
- 保存生成 metadata 和 SHA256；约定前 10,000 个为 paper-scale 子集。
- 生成后不可因某个模型结果不好而重抽测试集。
- CVRP100 继续使用官方 `vrp100_test_seed1234.pt`，不要重新生成替换。
- 新增 `generate_pomo_cvrp_testset.py`，只调用官方 `CVRProblemDef.get_random_problems()`，按 loader 需要保存 `depot_xy`、`node_xy`、`node_demand`。
- CVRP20/CVRP50 可各生成 10,000 个 seed1234 固定 instances，标明是我们生成的同分布集合，不是论文原始集合。
- 为 CVRP fixed set 做 round-trip：保存后由 `CVRPEnv.use_saved_problems()` 读取，并逐 tensor `torch.equal` 验证。

### Phase E: 完善固定 evaluator

- 新增 CVRP fixed evaluator，当前主输出为 x8 augmentation score。
- TSP/CVRP 最好保存每个 instance 的 x8 cost，汇总 CSV 再由独立脚本计算均值，便于以后与新 idea 做 paired comparison。
- single/no-aug 可以作为诊断列；如果保留 single，必须符合随机起点定义并固定随机种子，否则明确标记为非正式诊断值。
- 输出汇总还要包含 episodes、batch、aug factor、runtime、checkpoint、testset hash、环境。
- 先在 10 个手工/可检查 instances 上验证张量索引和候选取最优逻辑。

### Phase F: 正式 inference

- 用官方 checkpoint 跑 TSP20/TSP50/TSP100 的固定 10k 子集和 100k master。
- 用官方 checkpoint 跑 CVRP100 官方 10k fixed test set。
- 正式输出论文最后一行对应的 POMO x8 augmentation Len./score 与服务器 runtime。
- single/no-aug 若顺手输出，只作为附加诊断，不扩大当前交付范围。
- 与论文数值只做合理范围核对，并解释测试集、代码版本和硬件差异。
- CVRP20/CVRP50 没有官方 checkpoint，先在结果表标记 `deferred: train after primary reproduction`。

### Phase G: 已确认的低优先级工作

- 完成其他复现工作后，可以训练 CVRP20/CVRP50。
- 训练前必须从论文、OLD 原始代码及可验证材料补齐原文训练环境；不要只把 CVRP100 的 epoch 和 scheduler 生搬硬套到 N=20/50。
- 训练配置、随机种子、训练曲线、checkpoint 和环境版本都必须保存。
- Table 2/3 前面的外部 baseline 不复现。
- 当前不为 Gap 单独复现整表 baseline。以后如确需 Gap，只在同一个固定测试集上建立 reference costs。
- TSP 正式结果同时保留论文 10k 口径和 NEW 版 100k 口径，并明确命名，避免混淆。

## 14. 预期交付物

服务器 Codex 完成第一阶段后，请返回：

1. 仓库 commit 和 dirty status。
2. 服务器环境清单。
3. checkpoint/testset inventory。
4. 修改后脚本 diff，重点说明 x8 候选聚合、CUDA timing、fixed loader，以及如何处理非必需的 single trajectory 列。
5. smoke test 命令、完整终端关键输出和 CSV。
6. 固定测试集文件名、shape、seed、generator、SHA256。
7. 正式 POMO x8 augmentation 结果表；single/no-aug 仅在已正确实现时作为附加诊断。
8. 与论文目标值的差异及可证实原因，不得猜测。
9. CVRP20/50 低优先级训练前的原文配置审计结果，以及尚未具备的材料。

## 15. 给服务器 Codex 的第一条行动指令

先不要训练，也不要启动 full inference。请先只完成 Phase A 和 Phase B：审计服务器 POMO 仓库与 Python/CUDA 环境，检查我传来的三个脚本，列出你发现的路径/兼容性/实现问题，并提出最小修正 diff。确认 smoke test 方案后再执行 Phase C。整个过程中不得覆盖学长现有改动，也不得把新生成测试集称为论文作者原始测试集。

## 16. 从本机复制文件到服务器

在本机 PowerShell 执行。将 `<服务器别名>` 替换成 `C:\Users\84032\.ssh\config` 中的 `Host`，将 `/服务器上的/POMO` 替换成学长仓库的真实绝对路径。不要在命令或聊天中填写密码；`scp` 会在需要时单独提示输入。

先在服务器建立独立目录：

```powershell
ssh <服务器别名> "mkdir -p /服务器上的/POMO/reproduction_tools /服务器上的/POMO/reproduction_records"
```

复制三个脚本：

```powershell
scp "C:\Users\84032\Documents\Codex\2026-07-09\pomo-pomo-baseline-vrp-tsp-pomo\work\run_pomo_reproduction.py" `
    "C:\Users\84032\Documents\Codex\2026-07-09\pomo-pomo-baseline-vrp-tsp-pomo\work\generate_pomo_tsp_testset.py" `
    "C:\Users\84032\Documents\Codex\2026-07-09\pomo-pomo-baseline-vrp-tsp-pomo\work\run_pomo_tsp_fixed_eval.py" `
    <服务器别名>:/服务器上的/POMO/reproduction_tools/
```

复制交接文件和 smoke 记录：

```powershell
scp "C:\Users\84032\Documents\Codex\2026-07-09\pomo-pomo-baseline-vrp-tsp-pomo\SERVER_CODEX_HANDOFF.md" `
    "C:\Users\84032\Documents\Codex\2026-07-09\pomo-pomo-baseline-vrp-tsp-pomo\outputs\pomo_reproduction_smoke.csv" `
    "C:\Users\84032\Documents\Codex\2026-07-09\pomo-pomo-baseline-vrp-tsp-pomo\outputs\pomo_tsp20_fixed_tiny.csv" `
    <服务器别名>:/服务器上的/POMO/reproduction_records/
```

tiny `.pt` 仅在需要验证 fixed loader 时复制：

```powershell
scp "C:\Users\84032\Documents\Codex\2026-07-09\pomo-pomo-baseline-vrp-tsp-pomo\outputs\fixed_testsets\tsp20_test_seed1234_tiny.pt" `
    <服务器别名>:/服务器上的/POMO/reproduction_records/
```

传输后在服务器执行：

```bash
cd /服务器上的/POMO
sha256sum reproduction_tools/*.py
```

如果学长的 POMO 目录不允许写入，就把 `reproduction_tools` 和 `reproduction_records` 建在你自己的 home 目录，并在运行时使用 `--pomo-root` 指向只读的 POMO 仓库。
