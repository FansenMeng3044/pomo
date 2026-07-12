# POMO giant tour: raw-tour reward for training, Split only at evaluation

This variant is a sibling of `POMO_SPLIT`. The policy network, encoder/decoder,
and POMO training loop are identical. **The only difference is the reward.**

| Phase | Reward / score |
|-------|----------------|
| **Training** | Raw giant-tour length `depot -> pi[0] -> ... -> pi[-1] -> depot`. Capacity and demand are ignored; the tour is **never split**. (Equivalent to Split with infinite capacity.) |
| **Evaluation** | Optimal hard-capacity Bellman **Split** cost of the same permutation — a feasible CVRP objective. |

The switch is the `split_reward` env param on `GiantTourEnv`:

* `GiantTourTrainer` builds the env with `split_reward=False` (raw tour length).
* `GiantTourTester` forces `split_reward=True` (feasible CVRP cost via Split).

So the policy learns to produce a short *single* tour, and at test time we ask
"how good is a CVRP solution if we optimally split that tour under capacity?"

## Files

- `GiantTourEnv.py` — env with the two reward modes + fixed-test-set loading.
- `GiantTourModel.py` — POMO policy (identical to `POMO_SPLIT`).
- `SplitDecoder.py` — Bellman Split (used only at evaluation).
- `GiantTourTrainer.py` — REINFORCE with shared POMO baseline.
- `train_n100.py` — training entry (raw-tour reward).
- `GiantTourTester.py` / `test_n100.py` — evaluation entry (Split scoring).

## Run

CPU smoke test of training:

```console
python train_n100.py --smoke
```

Full CVRP100 training (raw-tour reward):

```console
python train_n100.py
```

Evaluate a trained checkpoint by splitting its tours (edit `model_load` in
`test_n100.py` to point at your checkpoint):

```console
python test_n100.py
```

The evaluation uses the official fixed test set `../vrp100_test_seed1234.pt`,
`model.eval()` argmax decoding, and x8 augmentation, so its `NO-AUG SPLIT SCORE`
and `AUGMENTATION SPLIT SCORE` are directly comparable to the official POMO
CVRP100 numbers and to the `POMO_SPLIT` model.
