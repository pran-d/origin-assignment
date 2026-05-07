# Task 1 : BC Training

## Architecture

Used a 4-layer MLP (obs_dim → 128 → 128 → 128 → act_dim) with ReLU activations and a Tanh output. The Tanh output hard-constrains predictions to [-1, 1], matching the action space without any clipping post-hoc. Instead of the suggested architecture of two ReLU hidden layers of 256 features, I used three ReLU hidden layers with 128 features. Three hidden layers are sufficient for this low-dimensional task (19-dim obs, 7-dim action); adding more layers would risk overfitting to the small number of data samples. A reduced hidden size with more depth is used because it gives a better success rate, probably explained through better hierarhical feature learning and reduced overfitting. 

## Loss function

MSE on continuous actions. The dataset is expert-only (lift-ph = proficient human), so fitting to MSE should work. To help with generalization and prevent overfitting to the training data, I use regularization (on the weights of the model). 

## Training duration

A training duration of 20 epochs is selected. The training/validation MSE flattens after the mid-teens, signalling a plateau in learning: after high-level features are learned, a few more epochs sharpen finer details. Beyond ~20 epochs, train loss continues to creep down while val loss stalls — a sign of overfitting on this small dataset.

## Stopping criterion

Fixed epoch count with an 80/20 train/val split *by trajectory* (not by transition — splitting by transition would leak demos across sets). Both train and val MSE are logged each epoch; the final 20 was chosen so val loss is at/near its minimum without diverging from train.



# Task 2 : BC failure-mode investigation

The BC policy achieves ~90% success. The remaining failures are identified to have the following features by inspecting per-rollout observation trajectories:
- The robot successfully reaches the cube but then the lift action saturates near zero because it is unable to grasp the cube. 
- This could happen because BC averages over the entire action distribution — for high-precision tasks like grasping, we need extremely accurate control actions.
- A slight error would in turn make the next observations out of distribution for the model, thus compounding into a failure very easily.

**Out-of-distribution state compounding (covariate shift):** BC is trained on expert demonstrations, so the observation distribution during training is the expert's trajectory distribution. At test time, any small deviation from an expert trajectory — caused by BC's own prediction error — produces an observation that was never seen during training. BC then receives an out-of-distribution (OOD) input, makes a potentially larger error, which shifts the observation further from the training manifold, and so on. This covariate shift (DAgger, Ross & Bagnell 2011) is the root cause of BC's compounding failures: even a 90% per-step accuracy can compound to 0% success over a long horizon if OOD states are badly handled. The NN-distance analysis confirms this: failure rollout observations have a measurably larger mean distance to the nearest training observation compared to success rollouts, showing that BC failures coincide with OOD states.

**Diagnostic method:** Logged `gripper_to_cube_pos` norm, gripper_qpos, and action z-component across all 50 rollouts. Failed rollouts were split by whether the min `gripper_to_cube_pos` norm ever went below 0.05 m. Additionally, nearest-neighbour L2 distance from each rollout observation to the training set was computed; failure rollouts show a heavier tail at large NN distances, confirming the OOD drift.

The residual's per-state correction is well-suited to both failure modes — it can learn to push the EEF closer during approach and inject upward velocity post-grasp, without touching the well-performing majority of the BC trajectory. Crucially, because the residual is conditioned on `a_BC(s)`, it can identify and correct states where BC is about to make an OOD-induced error before executing the bad action.



# Task 3 : Residual policy

## Architecture

`δ(s, a_BC(s))` — the delta net takes the concatenation of normalized obs and the BC action. Giving the residual access to `a_BC(s)` lets it condition on what BC is about to do rather than rediscovering it from obs, thus simplifying the learning process. In practice this means the residual can learn "when BC outputs a near-zero z-velocity during lift, add +δz". The delta net is a 2-layer MLP with 128 hidden units — small enough to be fast and to resist overfitting on the small dataset.

## Activation on delta

`tanh × DELTA_BOUND`. Tanh provides a smooth, differentiable bound that saturates gracefully, keeping gradients non-zero throughout training. Hard-clipping this would create a discontinuity that can harm the actor update whenever the delta saturates (which can happen often early in training). Instead, a soft-constraint (L2 penalty on the delta magnitude) would act as a regularizer, but also requires tuning an extra hyperparameter.

## Bound magnitude

DELTA_BOUND was calibrated from the data: computed `|a_demo - a_BC(s)|` across all state transitions and took the **90th percentile**. The intuition: the residual should be large enough to cover the typical BC error but not so large that it can override BC entirely. The 90th percentile captures "large but not outlier" errors. The chosen bound should also be sufficiently smaller than the absolute range of actions in the demo dataset.

## Algorithm

TD3+BC. The actor loss is:

```
L_actor = - λ * Q(s, a_executed) + MSE(δ_θ(s), clamp(a_demo - a_BC(s), ±DELTA_BOUND))
```

where `λ = ALPHA_BC / (mean(|Q|) + ε)` (with `ALPHA_BC = 2.5`) normalizes the Q scale so the RL term magnitude stays roughly constant as Q values evolve. The unweighted BC-regularization term keeps the delta anchored to the oracle residual computed from the demonstration data. Without this anchor, the actor would degenerate into a pure RL policy driven by a noisy offline Q-function: empirically `delta_mag` saturates at `DELTA_BOUND` every step, overriding BC and dropping success significantly. The TD3+BC λ-normalization here strikes the balance — the BC term dominates early when Q is uncalibrated, and the Q gradient gradually pushes targeted corrections only where the data supports them.

The BC term trains the delta head to predict the oracle residual (the actual correction needed at each state). The oracle target is clamped to `±DELTA_BOUND` because anything beyond that range will not be physically realizable by the residual head.

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

# Task 5 : Final eval : Why Residual Policy + Shield?
With a functioning Q gradient the residual can learn to push the EEF closer during approach (correcting mode-1 failures) and inject upward z-velocity post-grasp (correcting mode-2 lift-stall failures) — both targeted corrections that a pure BC delta head cannot pick up from MSE alone.
