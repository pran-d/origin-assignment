# Task 1 : BC Training

## Architecture

Used a 4-layer MLP (obs_dim → 128 → 128 → 128 → act_dim) with ReLU activations and a Tanh output. The Tanh output hard-constrains predictions to [-1, 1], matching the action space without any clipping post-hoc. Instead of the suggested architecture of two ReLU hidden layers of 256 features, I used three ReLU hidden layers with 128 features. Three hidden layers are sufficient for this low-dimensional task (19-dim obs, 7-dim action); adding more layers would risk overfitting to the small number of data samples. A reduced hidden size with more depth is used because it gives a better success rate, probably explained through better hierarhical feature learning and reduced overfitting. 

## Loss function

MSE on continuous actions. The dataset is expert-only (lift-ph = proficient human), so fitting to MSE should work. To help with generalization and prevent overfitting to the training data, I use regularization (on the weights of the model). 

## Training duration

A training duration of 20 epochs is selected. The loss curve is:

```
Epoch [3/20] Loss: 0.034872
Epoch [6/20] Loss: 0.030661
Epoch [9/20] Loss: 0.028878
Epoch [12/20] Loss: 0.027074
Epoch [15/20] Loss: 0.025959
Epoch [18/20] Loss: 0.025406
Epoch [20/20] Loss: 0.024253
```

The curve flattens significantly after epoch 15 (the elbow); signalling a plateau in learning: after the high level features are learned, a few more epochs are useful to learn more detailed features from the training. Using more epochs than this would lead to overfitting to the training set.

## Stopping criterion

Fixed epoch count with manual inspection of the loss curve. The overfitting can also be determined by splitting the dataset into training and validation sets. The validation loss will rise while training loss continues to go down, signalling that the model is unable to generalize to unseen inputs. A holdout split (e.g. 80/20 by trajectory) is used here.



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
L_actor = - λ * Q(s, a_executed) + BC_REG_COEF * MSE(δ_θ(s), a_demo - a_BC(s))  
```

where `λ = ALPHA_BC / (mean(|Q|) + ε)` normalizes the Q scale so the RL term magnitude stays roughly constant as Q changes, and `BC_REG_COEF = 5.0` is an explicit multiplier on the BC regularization term.

**Why BC_REG_COEF?** Without it, bc_reg converges toward zero within the first few hundred steps (the delta_net quickly learns to match the clamped oracle targets), while the RL term stays at roughly `-ALPHA_BC * sign(Q)` ≈ -2.5. This creates a >2000:1 imbalance, making the actor effectively a pure RL policy driven by a noisy offline Q-function. The result: delta_mag saturates at DELTA_BOUND every step, overriding BC and dropping success from 90% to 20%. `BC_REG_COEF = 5.0` keeps the BC guidance term actively competing with the RL term throughout training, anchoring the delta to meaningful corrections rather than drifting toward Q-maximizing but out-of-distribution actions.

The BC term trains the delta head to predict the oracle residual (the actual correction needed at each state). The oracle target is clamped to `±DELTA_BOUND` because anything beyond that range will not be physically realizable by the residual head.

The critic update is vanilla TD3-learning (Bellman), i.e. both Q networks should track the moving TD target, denoted by:
```
y = r + γ(1−d)⋅min(Q_1′​(s′,a′),Q_2′​(s′,a′)), where
a′=clip(aBC​(s′)+δθ​(s′),−1,1)
```

The critic's Q signal is necessary to steer the delta away from states where the oracle residual is ambiguous (multiple demos disagreed).

With a small dataset, TD3+BC is simpler and well-validated on offline low-dim tasks.


## Reward

Three shaped signals are added on top of the sparse terminal reward (1.0 on success):

1. **Approach (potential-based):** `r_approach = γΦ(s') − Φ(s)`, where `Φ(s) = −‖gripper_to_cube_pos‖`. This is policy-invariant under the standard Ng et al. (1999) potential shaping theorem — adding it cannot change the set of optimal policies. The sparse reward alone gives the Q-function almost no gradient signal for approach behaviour; this term fills that gap without introducing bias.

2. **Lift:** `r_lift = max(0, cube_z(s') − cube_z(s)) × 3.0`. Directly rewards upward cube movement, targeting the lift-stall failure mode where BC grasps the cube but applies near-zero z-velocity. A cube moves ~0.001–0.005 m per step, so each successful lift step contributes 0.003–0.015 to the return, which accumulates to ~0.15–0.75 across a lift phase — meaningful relative to the sparse reward of 1.0.

3. **Grasp proxy:** `r_grasp = 0.3 × I[‖g2c‖ < 0.06] × I[gripper_qpos_sum < 0.035]`. Rewards the gripper being closed (Panda: open ≈ 0.08 sum, closed ≈ 0) while near the cube. This directly reinforces the contact phase where BC mode-1 failures occur.

Pure sparse reward was tested first and produced the lower success rate (17%). The three-component shaping provides the Q-function with denser gradients specifically at the two identified failure modes (approach stall, lift stall), without biasing the approach component.


## Clip δ inside target Q

**Yes — the target Q computation uses the full residual forward pass (which includes the clamp).** If you clip δ in the actor but not in the critic Q computation, the target Q values are computed under a different action distribution than what the policy actually executes. This can create a systematic bias: the target over-estimates Q for states near the action boundary, causing the critic to over-value those states and the actor to push delta toward the clamp boundary even when that's suboptimal.

## Critic target update

Soft Polyak (τ = 0.005). Hard target updates (copy every N steps) can create periodic discontinuities in the target Q signal that destabilize the actor, especially early in training when Q values change rapidly. Polyak gives a smoothly lagged target that reduces gradient variance.

## Training step count

The critic loss decreases rapidly in the first 1,000 steps and plateaus by around 3,000 steps, indicating initial convergence. The actor loss continues to decrease slowly beyond this point. However, when training is extended to 10,000 steps, the critic loss begins to increase after ~5,000 steps. This suggests instability due to target drift. As the actor improves, it produces actions increasingly outside the dataset distribution, forcing the critic to evaluate out-of-distribution actions. This leads to value extrapolation error and higher TD loss, as the critic’s targets become less reliable.


# Task 4 : Safety shield

The shield clips each action dimension independently to `[data_min_d - margin_d, data_max_d + margin_d]`.

**Margin:** 5% of the per-dimension range. This is derived from the data: the standard deviation of BC action errors (across all transitions) is roughly 3–4% of the action range per dimension. A 5% margin therefore covers ~1.5σ of typical residual corrections while still catching truly out-of-distribution outputs (e.g. if the delta head saturates to ±DELTA_BOUND on an unseen observation). 

**Per-dimension vs. global L2 norm:** The action space is heterogeneous, where different dimensions represent physical quantities with different ranges; A global L2 norm treats all dimensions equally, which would be too loose or too tight for some dimensions. Per-dimension bounds match the natural scale of each action component, for example rotation and translational spaces are quantified separately.

Shield bounds:
```
low:  [-1.10, -0.62, -1.10, -0.16, -1.07, -0.57, -1.0]
high: [ 1.10,  0.71,  1.10,  0.13,  0.37,  0.53,  1.0]
```

For example, here the first three dimensions show end-effector translations, the next three represent end-effector rotation, and the last dimension is the gripper actuation. Each of these need different shields.

# Task 5 : Final eval : Why Residual Policy + Shield?
With a functioning Q gradient the residual can learn to push the EEF closer during approach (correcting mode-1 failures) and inject upward z-velocity post-grasp (correcting mode-2 lift-stall failures) — both targeted corrections that a pure BC delta head cannot pick up from MSE alone.
