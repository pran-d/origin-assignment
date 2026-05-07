# Notes 
## Suggested improvements

### Notebook
- **Validation-loss-based early stopping for BC.** You already log val loss but train for a fixed 20 epochs. A small change: track best val loss, save the checkpoint at the best epoch, and reload it after the loop. Removes the manual "20 looks fine" heuristic.
- **Seed the env explicitly per rollout.** `evaluate()` seeds numpy/torch once at start but `make_env()` doesn't take a seed. Identical seeds → identical rollouts is currently fragile; explicitly call `env.reset(seed=...)` (or robosuite's equivalent) per rollout to make the BC-vs-Residual comparison genuinely paired.
- **Diagnostic NN-distance is O(N·M) brute force.** For 9,666×~7,000 with 19 dims it works on a T4, but a `sklearn.neighbors.NearestNeighbors(algorithm='ball_tree')` or torch-on-GPU chunked version would be 10–50× faster and scales if the dataset grows.
- **Pre-stack diagnostic actions.** `np.stack([dataset[i]["action"].numpy() for i in range(len(dataset))])` in the shield cell rebuilds the array element-by-element; just use `dataset.actions.numpy()` directly.
- **Render flag dead code.** `run_rollout` accepts `render=True` but `evaluate` always passes `render = ... and False`, so videos are never actually saved (both `Video()` calls would point at empty/non-existent files). Either fix the flag or remove the rendering branch + video-related plumbing.
- **DataLoader `num_workers=0` + repeated iter recreation.** For 9k samples it's fine, but a single `for batch in cycle(loader)` (via `itertools.cycle`) is cleaner than the manual `try/except StopIteration` pattern.
- **Twin-critic asymmetry on actor side.** Standard TD3 uses only `Q1` for the actor's policy gradient (`q1_pi.mean()`), not `min(q1, q2)`. Using the min on both critic update *and* actor update biases the policy gradient pessimistically. Worth ablating.
- **Target policy smoothing missing.** TD3's third trick — Gaussian noise on the target action before computing target Q — isn't implemented. Adding `next_action += clamp(noise, -c, c)` (then re-clipping) would harden the critic against exploitation of sharp Q peaks.
- **Delayed actor updates.** Vanilla TD3 updates the actor every 2 critic steps. Currently they're updated on the same cadence; switching to delayed updates often reduces actor-loss oscillation in offline settings.

### Writeup
- **Quantify the failure-mode claims.** The writeup says "failure rollouts have a measurably larger mean NN distance" — give the actual numbers (median NN dist for success vs failure) since the notebook computes them. A single sentence with two numbers is much stronger than a qualitative claim.
- **Report shield clip rate.** The assignment explicitly asks for it as bonus ("how often does the shield actually trigger?"). Easy to add: count `np.any(action != shielded_action)` per step during the residual rollout.
- **Final eval numbers.** The Task 5 section ("Why Residual Policy + Shield?") doesn't actually report the headline BC-vs-Residual success rates and latency from the final eval cell. Add a small table.
- **Typo:** "hierarhical" → "hierarchical" in Task 1 Architecture.
- **Tighten Task 2.** The "Out-of-distribution state compounding (covariate shift)" paragraph repeats the bullets above it. Could be one paragraph instead of bullets-plus-paragraph.
