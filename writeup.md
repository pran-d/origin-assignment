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

The BC policy achieves 93% success, which is already very high. The remaining 7% failures are all identified to have the following features by inspecting per-rollout observation trajectories:
- The robot successfully reaches the cube but then the lift action saturates near zero because it is unable to grasp the cube. 
- This could happen because BC averages over the entire action distribution — for high-precision tasks like grasping, we need extremely accurate control actions. 
- A slight error would in turn make the next observations out of distribution for the model, thus compounding into a failure very easily. 

**Diagnostic method:** Logged `gripper_to_cube_pos` norm, gripper_qpos, and action z-component across all 50 rollouts. Failed rollouts diagnostics were plotted and they were split by whether the min `gripper_to_cube_pos` norm ever went below 0.05 m (grasped) or not (missed). Of the 2 failures, both were below the threshold.

 The residual's per-state correction is well-suited to any mode of failure — it can learn to push the EEF closer during approach and inject upward velocity post-grasp, without touching the well-performing majority of the BC trajectory.



# Task 3 : Residual policy

## Architecture

`δ(s, a_BC(s))` — the delta net takes the concatenation of normalized obs and the BC action. Giving the residual access to `a_BC(s)` lets it condition on what BC is about to do rather than rediscovering it from obs, thus simplifying the learning process. In practice this means the residual can learn "when BC outputs a near-zero z-velocity during lift, add +δz". The delta net is a 2-layer MLP with 128 hidden units — small enough to be fast and to resist overfitting on the small dataset.

## Activation on delta

`tanh × DELTA_BOUND`. Tanh provides a smooth, differentiable bound that saturates gracefully, keeping gradients non-zero throughout training. Hard-clipping this would create a discontinuity that can harm the actor update whenever the delta saturates (which can happen often early in training). Instead, a soft-constraint (L2 penalty on the delta magnitude) would act as a regularizer, but also requires tuning an extra hyperparameter.

## Bound magnitude

DELTA_BOUND was calibrated from the data: computed `|a_demo - a_BC(s)|` across all state transitions and took the 90th percentile. This gives **DELTA_BOUND = 0.0422**. The intuition: the residual should be large enough to cover the typical BC error but not so large that it can override BC entirely. The 90th percentile captures "large but not outlier" errors. The chosen bound should also be sufficiently smaller than the absolute range of actions in the demo dataset.

## Algorithm

TD3+BC. The actor loss is:

```
L_actor = - Q(s, a_executed) + λ * MSE(δ_θ(s), a_demo - a_BC(s))  
```

where `λ = ALPHA_BC / (mean(|Q|) + ε)` normalizes the Q scale so the RL and BC terms stay balanced as Q changes throughout training. The BC term trains the delta head to predict the oracle residual (the actual correction needed at each state). The oracle target is clamped to `±DELTA_BOUND` because anything beyond that range will not make physical sense to add.

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
