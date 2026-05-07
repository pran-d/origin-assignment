# Task 1 : BC Training

## Architecture

Used a 4-layer MLP (obs_dim → 128 → 128 → 128 → act_dim) with ReLU activations and a Tanh output. The Tanh output hard-constrains predictions to [-1, 1], matching the action space without any clipping post-hoc. Instead of the suggested architecture of two ReLU hidden layers of 256 features, I used three ReLU hidden layers with 128 features. Three hidden layers are sufficient for this low-dimensional task (19-dim obs, 7-dim action); adding more layers would risk overfitting to the small number of data samples. A reduced hidden size with more depth is used because it gives a better success rate, probably explained through better hierarchical feature learning and reduced overfitting. 

## Loss function

MSE on continuous actions. The dataset is expert-only (lift-ph = proficient human), so fitting to MSE should work. To help with generalization and prevent overfitting to the training data, I use regularization (on the weights of the model). 

## Training duration

A training duration of 20 epochs is selected. The training/validation MSE flattens after the mid-teens, signalling a plateau in learning: after high-level features are learned, a few more epochs sharpen finer details. Beyond ~20 epochs, train loss continues to creep down while val loss stalls — a sign of overfitting on this small dataset.

## Stopping criterion

Fixed epoch count with an 80/20 train/val split *by trajectory* (not by transition — splitting by transition would leak demos across sets). Both train and val MSE are logged each epoch; the final 20 was chosen so val loss is at/near its minimum without diverging from train.



# Task 2 : BC failure-mode investigation

The BC policy achieves **47/50 = 94%** success on the 50-rollout diagnostic (and 29/30 ≈ 96.7% on the 30-rollout final eval). The few failures share a clear pattern observed in the per-rollout `‖gripper_to_cube_pos‖` traces: the robot reaches the cube, but then the lift action saturates near zero because it fails to grasp it firmly. BC averages over the demo action distribution; high-precision grasping needs extremely accurate control, and even a small error puts the next obs slightly off the demo manifold, compounding into failure.

**Out-of-distribution state compounding (covariate shift):** BC is trained on expert demonstrations, so the observation distribution during training is the expert's trajectory distribution. At test time, any small deviation from an expert trajectory — caused by BC's own prediction error — produces an observation that was never seen during training. BC then receives an out-of-distribution (OOD) input, makes a potentially larger error, which shifts the observation further from the training manifold, and so on. This covariate shift (DAgger, Ross & Bagnell 2011) is the root cause of BC's compounding failures.

**Quantitative confirmation.** Per-state nearest-neighbour L2 distance from rollout obs to the training-set obs:
- Success rollouts — median NN distance **0.076**
- Failure rollouts — median NN distance **1.800** (≈ **23.7×** further from the demo manifold)

That ratio is a clean signature of covariate shift: failure trajectories drift to states the BC has effectively never seen.

**Diagnostic method:** Logged `gripper_to_cube_pos` norm, gripper_qpos, and the action z-component across all 50 rollouts. Failure rollouts were split by whether the min `gripper_to_cube_pos` ever went below 0.05 m, and the per-feature distributions (`‖g2c‖`, `eef_pos z`, mean `gripper_qpos`) were compared between dataset, BC successes, and BC failures. Failures show a heavier tail at large `‖g2c‖` and atypical gripper closure profiles.

The residual's per-state correction is well-suited to both failure modes — it can learn to push the EEF closer during approach and inject upward velocity post-grasp, without touching the well-performing majority of the BC trajectory. Crucially, because the residual is conditioned on `a_BC(s)`, it can identify and correct states where BC is about to make an OOD-induced error before executing the bad action.



# Task 3 : Residual policy

## Architecture

`δ(s, a_BC(s))` — the delta net takes the concatenation of normalized obs and the BC action. Giving the residual access to `a_BC(s)` lets it condition on what BC is about to do rather than rediscovering it from obs, thus simplifying the learning process. In practice this means the residual can learn "when BC outputs a near-zero z-velocity during lift, add +δz". The delta net is a 2-layer MLP with 128 hidden units — small enough to be fast and to resist overfitting on the small dataset.

## Activation on delta

`tanh × DELTA_BOUND`. Tanh provides a smooth, differentiable bound that saturates gracefully, keeping gradients non-zero throughout training. Hard-clipping this would create a discontinuity that can harm the actor update whenever the delta saturates (which can happen often early in training). Instead, a soft-constraint (L2 penalty on the delta magnitude) would act as a regularizer, but also requires tuning an extra hyperparameter.

## Bound magnitude

DELTA_BOUND was calibrated from the data: computed `|a_demo - a_BC(s)|` across all state transitions and took the **90th percentile**. The intuition: the residual should be large enough to cover the typical BC error but not so large that it can override BC entirely. The 90th percentile captures "large but not outlier" errors. The chosen bound should also be sufficiently smaller than the absolute range of actions in the demo dataset.

## Algorithm

TD3+BC with two stabilizers added on top of the textbook recipe — a BC-anchor multiplier and obs-noise augmentation. The actor loss is:

```
L_actor = - λ * Q1(s̃, a_exec(s̃)) + BC_REG_COEF * MSE(δ_θ(s̃), clamp(a_demo - a_BC(s), ±DELTA_BOUND))

s̃ = s + ε,  ε ~ N(0, (OBS_NOISE_STD · OBS_STD)²)
λ = ALPHA_BC / (mean(|Q1|) + ε)
ALPHA_BC = 0.25,  BC_REG_COEF = 15.0,  OBS_NOISE_STD = 0.10
```

**Why a `BC_REG_COEF` multiplier?** The λ-normalization keeps `|−λ·Q1|` roughly constant in magnitude, so once the delta head fits the oracle and `bc_reg → 0`, the unweighted BC term cannot compete with the constant-magnitude RL pull. We measured this directly in early ablations: dropping the explicit anchor saturates `delta_mag` at `DELTA_BOUND` and collapses success to 0/30. Setting `BC_REG_COEF = 15.0` keeps the BC-regression gradient dominant on-data so δ stays close to the oracle residual; `ALPHA_BC = 0.25` retains a small Q-driven nudge for the few states where multiple demos disagree.

**Why obs-noise augmentation?** Section 1.1 showed failure-rollout obs are ~24× further from the demo manifold than success-rollout obs. Training the residual only on clean demo obs leaves it brittle to that drift. We perturb the obs fed to the *actor* by Gaussian noise scaled to `0.10 · OBS_STD` per dim — this is DAgger-style augmentation, except we synthesize the OOD obs from the dataset rather than rolling out the policy. The critic update and the oracle target use clean obs so targets stay correct; only the actor's input is noised. Empirically this is the change that took the residual from regressing BC to matching it on the same paired-seed eval.

**TD3 tricks added:** Q1-only actor gradient (using `min(q1, q2)` on both critic *and* actor sides double-counts pessimism), target-policy smoothing (clipped Gaussian noise on the next-action before the target Q computation), and delayed actor + target updates every 2 critic steps. These are textbook TD3 stabilizers; ablating any of them did not flip the eval result, but they reduce actor-loss oscillation across runs.

**Q1-only on the actor, twin-min on the critic.** The critic update uses `min(Q1', Q2')` for the TD target (the standard TD3 anti-overestimation trick). The actor gradient uses Q1 alone — the twin networks are there to denoise the *bootstrap target*, not to apply a second layer of pessimism to the policy improvement.

The oracle target is clamped to `±DELTA_BOUND` because anything beyond that range cannot be produced by the bounded residual head.

The critic update is vanilla TD3-learning (Bellman), i.e. both Q networks should track the moving TD target, denoted by:
```
y = r + γ(1−d)⋅min(Q_1′​(s′,a′),Q_2′​(s′,a′)), where
a′=clip(aBC​(s′)+δθ​(s′),−1,1)
```

The critic's Q signal is necessary to steer the delta away from states where the oracle residual is ambiguous (multiple demos disagreed).

With a small dataset, TD3+BC is simpler and well-validated on offline low-dim tasks.


## Reward

Sparse terminal reward as provided (1.0 on success, 0 otherwise). Shaped rewards (e.g. `-|cube - eef|`) were considered but rejected: the vanilla BC is able to achieve a very high success rate, so sparse signals near the goal are sufficient. The BC model does not need additional guidance from RL to REACH the cube itself, it only needs it to GRASP it. Shaping introduces an additional hyperparameter (reward scale).


## Clip δ inside target Q

**Yes — the target Q computation uses the full residual forward pass (which includes the clamp).** If you clip δ in the actor but not in the critic Q computation, the target Q values are computed under a different action distribution than what the policy actually executes. This can create a systematic bias: the target over-estimates Q for states near the action boundary, causing the critic to over-value those states and the actor to push delta toward the clamp boundary even when that's suboptimal.

## Critic target update

Soft Polyak (τ = 0.005). Hard target updates (copy every N steps) can create periodic discontinuities in the target Q signal that destabilize the actor, especially early in training when Q values change rapidly. Polyak gives a smoothly lagged target that reduces gradient variance.

## Training step count

5,000 gradient steps. The critic loss decreases rapidly in the first ~1,000 steps and plateaus by ~3,000; the actor loss continues to decrease slowly beyond this point. When training is extended past 5,000, the critic loss begins to drift upward — an instability driven by target drift: as the actor improves, it produces actions increasingly outside the dataset distribution, forcing the critic to evaluate OOD actions and accruing value-extrapolation error. Stopping at 5,000 keeps the actor inside the regime where the critic's TD targets are trustworthy.

# Task 4 : Safety shield

The shield clips each action dimension independently to `[data_min_d - margin_d, data_max_d + margin_d]`.

**Margin:** 5% of the per-dimension range, with one exception. The standard deviation of BC action errors across all transitions is roughly 3–4% of the action range per dimension, so a 5% margin covers ~1.5σ of typical residual corrections while still catching truly out-of-distribution outputs (e.g. if the delta head saturates to ±DELTA_BOUND on an unseen observation).

**Exception — gripper (dim 6):** margin set to 0. The gripper is binary in practice (-1 open / +1 close); a non-zero margin would silently allow intermediate values that never appear in demos. Hence the gripper bounds stay at exactly `[-1, +1]`.

**Per-dimension vs. global L2 norm:** The action space is heterogeneous, where different dimensions represent physical quantities with different ranges; A global L2 norm treats all dimensions equally, which would be too loose or too tight for some dimensions. Per-dimension bounds match the natural scale of each action component, for example rotation and translational spaces are quantified separately.

Shield bounds:
```
low:  [-1.10, -0.62, -1.10, -0.16, -1.07, -0.57, -1.0]
high: [ 1.10,  0.71,  1.10,  0.13,  0.37,  0.53,  1.0]
```

For example, here the first three dimensions show end-effector translations, the next three represent end-effector rotation, and the last dimension is the gripper actuation. Each of these need different shields.

# Task 5 : Final eval

**Headline numbers (30 paired rollouts, seed 42):**

| Metric                  |    BC | Residual+Shield | Δ        |
| ----------------------- | ----: | --------------: | :------- |
| success_rate            | 0.967 |           0.933 | -0.033   |
| mean_steps_on_success   | 45.28 |           44.82 | -0.45    |
| p99 latency (ms)        |  0.92 |            1.48 | +0.56    |

**Shield clip rate (residual rollouts):** 30.4% of steps had at least one dimension clipped. Almost all of that is on dim 3 (rotation about y), with smaller activity on dims 4 and 5 — consistent with the residual occasionally over-rotating during approach. Translation dims (0,1,2) and the gripper (6) are never clipped, indicating the residual stays inside the demo-action support there.

**Parameter counts:** BC = 36,487; Residual (BC + δ-head) = 57,358 — δ adds ~21k params (~57% on top of BC).

**Why Residual Policy + Shield?** The Q gradient gives the residual a way to nudge actions toward higher-value behavior at states where multiple demos disagree — pushing the EEF closer during approach and adding a small z-bias post-grasp — without rewriting BC's bulk behavior. Obs-noise augmentation gives it robustness to the very OOD obs (NN-distance ratio 23.7×) that drive BC's failures. The shield is an inexpensive backstop: it costs essentially nothing on the latency budget and it catches any per-dim excursion the learned δ might still produce on a truly novel obs.

**Honest caveat.** On this run the residual is slightly *below* BC (0.933 vs 0.967, one fewer success). With BC already saturating the lift-ph success ceiling, the residual's marginal value here is small and within run-to-run variance on 30 rollouts. The case for the residual+shield architecture is the *worst-case* behavior on harder/longer-horizon tasks where BC alone won't reach 97%; on lift-ph specifically it primarily demonstrates that we can add a Q-shaped correction without breaking a strong BC backbone.
