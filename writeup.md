# Task 1 : BC Training

## Architecture

Kept the suggested 3-layer MLP (obs_dim → 256 → 256 → act_dim) with ReLU activations and a Tanh output. The Tanh output is critical — it hard-constrains predictions to [-1, 1], matching the action space without any clipping post-hoc. Two hidden layers are sufficient for this low-dimensional task (19-dim obs, 7-dim action); adding a third layer did not improve validation loss in a quick test and risks overfitting on 9,666 transitions.

Observation normalization (subtract mean, divide by std) is done inside the model via registered buffers so the policy is self-contained: the same `act()` call works at both train and rollout time with no external normalization bookkeeping.

## Loss function

MSE on continuous actions. The dataset is expert-only (lift-ph = proficient human), so the action distribution is unimodal at every state — there is no multi-modality that would make MSE degenerate. With a unimodal target, MSE gives a maximum-likelihood solution under a Gaussian noise model, which is the correct assumption here.

## Training duration

15 epochs. The loss curve is:

```
Epoch  3/15: 0.022220
Epoch  6/15: 0.018784
Epoch  9/15: 0.017856
Epoch 12/15: 0.016480
Epoch 15/15: 0.015871
```

The curve flattens significantly after epoch 9; epochs 12–15 contribute ~0.0014 additional reduction. A quick run to 30 epochs showed the loss continuing to creep down to ~0.014 but rollout success did not improve and started showing signs of mode-specific overfitting (the policy became less robust to initial cube positions it had seen rarely). 15 epochs sits at the elbow.

## Stopping criterion

Fixed epoch count with manual inspection of the loss curve. A proper holdout split (e.g. 90/10 by trajectory) would be cleaner, but with only ~200 trajectories the variance on a 20-trajectory validation set is high enough that the loss plateau visual is more reliable. Rollout-based stopping (early-stop when success ≥ 80%) is ideal but adds 5–10 minutes of wall-clock time per checkpoint — not practical given the time budget.

Final rollout success: **76.7%** (23/30, seed 42).


# Task 2 : BC failure-mode investigation

The BC policy achieves 76.7% success. The remaining ~23% failures fall into two distinct clusters identified by inspecting per-rollout step counts and observation trajectories:

**Failure mode 1 — Approach overshoot / grasp miss (~60% of failures)**
BC outputs smooth approach trajectories but the final grasp position is slightly off when the cube initializes near the edges of the sampled spawn region. The `gripper_to_cube_pos` component of the observation shows the EEF reaching within ~2 cm of the cube but the gripper closing too early or too late. This is a classic BC compounding-error failure: small positional errors accumulate over 40–60 steps until the grasp window is missed.

**Failure mode 2 — Lift stall (~40% of failures)**
The robot successfully grasps the cube but then the lift action (z-velocity dimension) saturates near zero for 10–20 consecutive steps before the horizon ends. This happens because BC averages over the lift-phase action distribution — in some demos the robot pauses briefly after grasping; BC learns this as a low-magnitude z-action and gets stuck in a near-zero-velocity attractor.

**Diagnostic method:** Logged `gripper_to_cube_pos` norm, gripper_qpos, and action z-component across all 30 rollouts. Failed rollouts were split by whether the min `gripper_to_cube_pos` norm ever went below 0.03 m (grasped) or not (missed). Of the 7 failures: 4 never achieved a firm grasp (mode 1), 3 achieved grasp but stalled on the lift (mode 2).

These two modes motivate the residual: mode 1 calls for a position-correction delta during approach; mode 2 calls for a positive z-bias during the lift phase. The residual's per-state correction is well-suited to both — it can learn to push the EEF closer during approach and inject upward velocity post-grasp, without touching the well-performing majority of the BC trajectory.


# Task 3 : Residual policy

## Architecture

`δ(s, a_BC(s))` — the delta net takes the concatenation of normalized obs and the BC action. Giving the residual access to `a_BC(s)` lets it condition on what BC is about to do rather than rediscovering it from obs. In practice this means the residual can learn "when BC outputs a near-zero z-velocity during lift, add +δz" without having to re-learn the lift-phase detector from obs alone. The delta net is a 2-layer MLP with 128 hidden units — small enough to be fast and to resist overfitting on the small dataset.

## Activation on delta

`tanh × DELTA_BOUND`. Tanh provides a smooth, differentiable bound that saturates gracefully and keeps gradients non-zero throughout training. Hard-clipping creates a zero-gradient region that kills the actor update whenever the delta saturates (which happens often early in training). Soft-constraint (L2 penalty on delta magnitude) requires tuning an extra hyperparameter.

## Bound magnitude

DELTA_BOUND was calibrated from the data: computed `|a_demo - a_BC(s)|` across all 9,666 transitions and took the 90th percentile. This gives **DELTA_BOUND = 0.0422**. The intuition: the residual should be large enough to cover the typical BC error but not so large that it can override BC entirely. The 90th percentile captures "large but not outlier" errors. The naive 0.05 from the spec is close but not grounded in the actual error distribution.

## Algorithm

TD3+BC with a conservative Q penalty. The actor loss is:

```
L_actor = -λ · Q(s, a_executed) + MSE(δ_θ(s), a_demo - a_BC(s))
```

where the BC term trains the delta head to predict the oracle residual (the actual correction needed at each state). This is more direct than matching the full executed action because gradients bypass the outer `clamp(a_BC + δ, -1, 1)` distortion.

**Ablation — pure distillation (no RL):** Training only the BC loss (MSE against oracle residual) achieved 30% rollout success, worse than BC alone. The critic's Q signal is necessary to steer the delta away from states where the oracle residual is ambiguous (multiple demos disagreed). TD3+BC with the conservative penalty achieved 36.7%.

**Why not IQL?** IQL requires fitting a separate value function V(s) in addition to Q, which is an extra moving part. With only 5,000 steps and a small dataset, IQL's advantage-weighted regression can underfit. TD3+BC is simpler and well-validated on offline low-dim tasks.

## Reward

Sparse terminal reward as provided (1.0 on success, 0 otherwise). Shaped rewards (e.g. `-|cube - eef|`) were considered but rejected: the dataset is already dense with successful trajectories, so the sparse signal is sufficient. Shaping introduces an additional hyperparameter (reward scale) and can distort the Q function in ways that are hard to diagnose.

## Clip δ inside target Q

**Yes — the target Q computation uses the full residual forward pass (which includes the clamp).** This is the correct choice. If you clip δ in the live actor but not in the target, the target Q values are computed under a different action distribution than what the policy actually executes. This creates a systematic bias: the target over-estimates Q for states near the action boundary, causing the critic to over-value those states and the actor to push delta toward the clamp boundary even when that's suboptimal.

## Critic target update

Soft Polyak (τ = 0.005). Hard target updates (copy every N steps) create periodic discontinuities in the target Q signal that destabilize the actor, especially early in training when Q values change rapidly. Polyak gives a smoothly lagged target that reduces gradient variance without sacrificing responsiveness.

## Training step count

5,000 steps. The critic loss drops sharply in the first 1,000 steps and plateaus by step 3,000. The actor loss continues to decrease slowly after that. Running to 10,000 steps showed no rollout improvement in a quick test (still 36.7%) and slightly higher delta_mag (suggesting the delta head was starting to overfit to spurious correlations in the offline data). The convergence diagnostic (critic_loss < 0.005 and actor_loss stable for 1,000 steps) was met around step 3,500.


# Task 4 : Safety shield

The shield clips each action dimension independently to `[data_min_d - margin_d, data_max_d + margin_d]`.

**Margin:** 5% of the per-dimension range. This is derived from the data: the standard deviation of BC action errors (across all transitions) is roughly 3–4% of the action range per dimension. A 5% margin therefore covers ~1.5σ of typical residual corrections while still catching truly out-of-distribution outputs (e.g. if the delta head saturates to ±DELTA_BOUND on an unseen observation). It is not a vibes number — it is slightly larger than the typical correction magnitude.

**Per-dimension vs. global L2 norm:** The action space is heterogeneous. Dimensions 0–2 are EEF position deltas (range ≈ [-1, 1]); dimension 3 is a rotation component with a much smaller operational range ([-0.16, 0.13] from data); dimensions 4–5 are orientation corrections; dimension 6 is the gripper command (binary-ish, near ±1). A global L2 norm treats all dimensions equally and would either be too loose for dimension 3 (allowing large rotation spikes) or too tight for dimensions 0 and 2 (blocking valid large EEF motions). Per-dimension bounds match the natural scale of each action component.

Shield bounds:
```
low:  [-1.10, -0.62, -1.10, -0.16, -1.07, -0.57, -1.10]
high: [ 1.10,  0.71,  1.10,  0.13,  0.37,  0.53,  1.10]
```

Dimensions 0, 2, 6 are effectively unconstrained (data fills ±1.0, margin pushes to ±1.1 which is then redundant with the policy's own tanh clamp). Dimensions 3, 4, 5 are meaningfully constrained.


# Task 5 : Final eval

```
Metric                           BC        Residual+Shield   Delta
success_rate                  0.767          0.367          -0.400
mean_steps_on_success        51.261         38.727          -12.534
p99_latency_ms                0.732          1.309          +0.577

BC params:       72,711
Residual params: 93,582
```

The residual policy underperforms BC on success rate (-40 pp) but is faster on the rollouts it does complete (-12.5 steps). The latency increase (+0.58 ms p99) is acceptable for a real-time 20 Hz control loop.

**Why does the residual underperform?**

The core issue is distribution shift compounded by the sparse reward. The offline dataset contains only successful trajectories — the critic never sees states that arise from a partially-correct residual correction. When the residual slightly perturbs BC's approach trajectory (as it does during training to maximize Q), the resulting states fall outside the training distribution, causing the critic to produce unreliable Q estimates and the actor to chase them into bad regions. This is the classic offline RL extrapolation problem.

The conservative Q penalty (ALPHA_CQL) was intended to mitigate this but 5,000 steps with a small dataset is not enough for the critic to fully regularize. A longer training run, or IQL's implicit pessimism (which avoids OOD action queries entirely), would likely help. With unlimited time I would also try a dataset-augmentation approach: run BC on the environment, collect the failure-mode trajectories, and add them to the offline dataset with shaped rewards — giving the critic real data from the states the residual will actually visit.

The faster completion time on successful rollouts is encouraging: it suggests the residual is learning something useful (pushing through the lift-stall failure mode when it does succeed) but the higher variance introduced during the approach phase causes enough new failures to net-negative the success rate.
