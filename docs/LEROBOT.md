# LeRobot Policy Analysis: Complete Algorithm Reference

## Table of Contents
1. [Common Programming Interface (Design Pattern)](#common-programming-interface)
2. [Policy Summaries by Category](#policy-summaries-by-category)
3. [Detailed Policy Analysis (17 Algorithms)](#detailed-policy-analysis)
4. [How to Create a New Policy](#how-to-create-a-new-policy)

---

## Common Programming Interface

All policies inherit from `PreTrainedPolicy` (in `src/lerobot/policies/pretrained.py`), which extends `nn.Module` + `HubMixin`.

### Required Class Attributes

```python
class MyPolicy(PreTrainedPolicy):
    config_class = MyPolicyConfig   # must point to your @register_subclass config
    name = "my_policy"              # string identifier used by --policy.type=
```

### Abstract Methods You Must Implement

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_optim_params` | `() -> dict` | Returns parameter groups for the optimizer (e.g., different LRs for backbone vs head) |
| `reset` | `() -> None` | Called when the environment resets. Clear action queues, caches, hidden states. |
| `forward` | `(batch: dict[str, Tensor]) -> tuple[Tensor, dict \| None]` | **Training**: takes a batch, returns `(loss, info_dict)`. The training loop calls this. |
| `predict_action_chunk` | `(batch, **kwargs) -> Tensor` | **Inference**: given observations, produce the full action chunk (before queue logic). |
| `select_action` | `(batch, **kwargs) -> Tensor` | **Inference**: return the single next action. Manages the action queue/cache internally. |

### Optional Methods to Override

| Method | Purpose |
|--------|---------|
| `_get_default_peft_targets() -> dict` | Return default LoRA target modules for PEFT fine-tuning |
| `_validate_peft_config(config)` | Policy-specific validation when wrapping with PEFT |

### Config Class Requirements

Your config must:
1. Inherit from `PreTrainedConfig` (in `src/lerobot/configs/__init__.py`)
2. Be decorated with `@PreTrainedConfig.register_subclass("my_policy")`
3. Be a `@dataclass` (parsed by draccus)
4. Define at minimum: `input_features`, `output_features`, normalization settings

### Factory Registration

In `src/lerobot/policies/factory.py`:
1. Import your config in the top-level imports
2. Add your policy to `get_policy_class()` (lazy import of modeling class)
3. Add your config to `make_policy_config()`
4. Add your processor factory to `make_pre_post_processors()`

### Processor Pipeline

Each policy needs a `make_<name>_pre_post_processors()` function that returns:
- A **preprocessor** (`PolicyProcessorPipeline`): normalizes observations before the model
- A **postprocessor** (`PolicyProcessorPipeline`): denormalizes predicted actions back to real space

### Typical Execution Flow

```
[Training]
  batch = dataset[i]
  batch = preprocessor(batch)
  loss, info = policy.forward(batch)
  loss.backward()

[Inference]
  obs = env.get_obs()
  obs = preprocessor(obs)
  action = policy.select_action(obs)
  action = postprocessor(action)
  env.step(action)
```

---

## Policy Summaries by Category

### Imitation Learning (Behavior Cloning)

| Policy | Generative Method | VLM Backbone | Paper Year |
|--------|------------------|--------------|------------|
| **Diffusion** | DDPM/DDIM denoising | None (ResNet encoder) | 2023 |
| **ACT** | CVAE + Transformer | None (ResNet encoder) | 2023 |
| **VQ-BeT** | VQ-VAE + GPT | None (ResNet encoder) | 2024 |
| **MultiTaskDiT** | Diffusion/Flow DiT | CLIP | 2025 |
| **PI0** | Flow Matching | PaliGemma (SigLIP+Gemma) | 2024 |
| **PI0-FAST** | Autoregressive tokens (DCT+BPE) | PaliGemma | 2025 |
| **PI0.5** | Flow Matching + AdaRMS | PaliGemma | 2025 |
| **SmolVLA** | Flow Matching | SmolVLM2 (SigLIP) | 2025 |
| **EO1** | Flow Matching | Qwen2.5-VL | 2024 |
| **XVLA** | Flow Matching (rectified flow) | Florence-2 | 2025 |
| **Wall-X** | Flow Matching + optional FAST | Qwen2.5-VL (MoE) | 2025 |
| **Groot** | Flow Matching (DiT) | Eagle2 (InternViT+LLM) | 2025 |
| **MolmoAct2** | Flow Matching + optional FAST | Molmo (ViT+Transformer) | 2026 |
| **VLA-JEPA** | Flow Matching (DiT) + JEPA world model | Qwen3-VL | 2026 |

### Reinforcement Learning

| Policy | Method | Category |
|--------|--------|----------|
| **TD-MPC** | Model-based RL + MPC (CEM/MPPI planning in latent space) | Offline RL / Offline-to-Online |
| **Gaussian Actor (SAC)** | Maximum-entropy off-policy actor-critic | Online off-policy RL |

### Inference-Only Enhancement

| Policy | Purpose |
|--------|---------|
| **RTC** | Real-Time Chunking — temporal consistency guidance for flow-matching policies at inference |


---

### Tier List & Recommendations

| Tier | Policy | Year | Recommendation | Reason |
|------|--------|------|----------------|--------|
| **SOTA (Current Best)** | **PI0.5**, **Groot**, **MolmoAct2**, **VLA-JEPA** | 2025-2026 | Must try for new projects | Strongest vision-language-action (VLA) models with Flow Matching + powerful VLMs. These are pushing real-world performance. |
| **Very Strong / Near SOTA** | **PI0**, **SmolVLA**, **EO1**, **Wall-X** | 2024-2025 | Highly recommended | Excellent balance of performance and efficiency. PI0 series is extremely influential. |
| **Foundational / Must-Read** | **Diffusion**, **ACT** | 2023 | **Must read & understand** | The two papers that reshaped the field in 2023. |
| **Important / Worth Reading** | **VQ-BeT**, **MultiTaskDiT**, **PI0-FAST** | 2024-2025 | Good to read | Solid contributions in tokenization or scaling. |
| **Specialized / Niche** | **TD-MPC**, **Gaussian Actor (SAC)**, **RTC** | - | Read if relevant | RL methods and inference tricks. |

---

## Detailed Policy Analysis

---

### 1. Diffusion Policy

**Paper**: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" — Chi et al., 2023 (arXiv:2303.04137)

https://arxiv.org/abs/2303.04137

https://diffusion-policy.cs.columbia.edu/

https://github.com/real-stanford/diffusion_policy

https://diffusion-policy.cs.columbia.edu/data/

https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing

https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing

**Category**: Imitation Learning (Behavior Cloning)

**Architecture**: ResNet18 vision encoder + SpatialSoftmax → 1D Convolutional U-Net with FiLM conditioning

**Training Objective**: MSE noise prediction (DDPM)

#### Formulas + Pseudo-code (side by side)

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Forward diffusion:                      │ noisy_traj = scheduler.add_noise(            │
│ x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε          │     trajectory, eps, timesteps)              │
│                                         │                                              │
│ Training loss:                          │ pred = unet(noisy_traj, t, global_cond)      │
│ L = E[‖ε_θ(x_t, t, c) - ε‖²]            │ loss = F.mse_loss(pred, eps)                 │
│                                         │                                              │
│ Inference (reverse):                    │ for t in scheduler.timesteps:                │
│ x_{t-1} = scheduler.step(ε_θ, t, x_t)   │   pred = unet(x_t, t, cond)                  │
│                                         │   x_t = scheduler.step(pred, t, x_t)         │
│                                         │                                              │
│ FiLM conditioning:                      │ scale, bias = Linear(Mish(cond))             │
│ out = scale · conv(x) + bias            │ out = scale * conv1(x) + bias                │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

**Training pseudo-code:**
```python
def forward(batch):
    img_features = rgb_encoder(batch["observation.images"])    # ResNet + SpatialSoftmax
    global_cond = flatten(concat(batch["observation.state"], img_features))
    trajectory = batch["action"]                               # (B, horizon, action_dim)
    eps = torch.randn_like(trajectory)
    t = torch.randint(0, num_train_timesteps, (B,))
    noisy_trajectory = scheduler.add_noise(trajectory, eps, t)
    pred = unet(noisy_trajectory, t, global_cond=global_cond)
    loss = F.mse_loss(pred, eps)
    return loss, {}
```

**Inference pseudo-code:**
```python
def select_action(batch):
    if action_queue is empty:
        global_cond = encode_observations(obs_queue)
        x = torch.randn(B, horizon, action_dim)
        for t in reversed(scheduler.timesteps):
            pred = unet(x, t, global_cond)
            x = scheduler.step(pred, t, x).prev_sample
        action_queue.extend(x[:, start:start+n_action_steps])
    return action_queue.popleft()
```

---

### 2. ACT (Action Chunking with Transformers)

**Paper**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" — Zhao et al., 2023 (arXiv:2304.13705)

https://arxiv.org/pdf/2304.13705

**Category**: Imitation Learning (CVAE-based Behavior Cloning)

**Architecture**: CVAE with Transformer encoder/decoder + ResNet18 vision backbone

**Training Objective**: L1 reconstruction + KL divergence

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Reparameterization:                     │ z = mu + exp(log_var/2) * randn_like(mu)     │
│ z = μ + σ · ε,  ε ~ N(0,I)              │                                              │
│                                         │                                              │
│ KL divergence:                          │ kl = -0.5 * sum(1 + log_var                  │
│ D_KL = -½ Σ(1 + log σ² - μ² - σ²)       │              - mu² - exp(log_var))           │
│                                         │                                              │
│ Total loss:                             │ loss = l1_loss + kl_weight * kl_loss         │
│ L = ‖a - â‖₁ + β · D_KL                 │                                              │
│                                         │                                              │
│ Temporal ensembling:                    │ w_i = exp(-m * i)                            │
│ a_t = Σ(w_i · a_i) / Σ(w_i)             │ action = weighted_sum / sum(weights)         │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

**Training pseudo-code:**
```python
def forward(batch):
    # VAE encode (training only)
    cls_out = vae_encoder([CLS, project(state), project(actions)])
    mu, log_var = split(linear(cls_out))
    z = mu + exp(log_var / 2) * torch.randn_like(mu)

    # Transformer encode observations + latent
    tokens = [project(z), project(state), *flatten(conv1x1(backbone(imgs)))]
    encoder_out = transformer_encoder(tokens)

    # Decode action chunk
    queries = torch.zeros(chunk_size, B, dim_model)
    decoder_out = transformer_decoder(queries, encoder_out)
    actions_hat = action_head(decoder_out)

    # Loss
    l1 = masked_l1(actions_hat, actions_gt)
    kl = (-0.5 * (1 + log_var - mu**2 - exp(log_var))).sum(-1).mean()
    return l1 + 10.0 * kl, {"l1": l1, "kl": kl}
```

**Inference pseudo-code:**
```python
def select_action(batch):
    z = torch.zeros(B, latent_dim)  # mode of prior at inference
    encoder_out = transformer_encoder([project(z), project(state), img_features])
    action_chunk = action_head(transformer_decoder(queries, encoder_out))
    # Either queue first n actions, or use temporal ensembling
    return next_action
```

---

### 3. VQ-BeT (Vector-Quantized Behavior Transformer)

**Paper**: "Behavior Generation with Latent Actions" — Lee et al., 2024 (arXiv:2403.03181)

https://arxiv.org/pdf/2403.03181v1

**Category**: Imitation Learning (Discretized Behavior Cloning)

**Architecture**: Residual VQ-VAE (action tokenizer) + minGPT (sequence model) + offset head

**Training Objective**: Two-phase: (1) VQ-VAE reconstruction, (2) Focal loss + L1 offset

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Phase 1 - VQ-VAE:                       │ z = encoder(action_flat)                     │
│ L₁ = ‖a - dec(VQ(enc(a)))‖₁             │ z_q, codes, vq_loss = RVQ(z)                 │
│   + 5 · commitment_loss                 │ loss = l1(a, decoder(z_q)) + 5*vq_loss       │
│                                         │                                              │
│ Phase 2 - GPT:                          │ logits = MLP_bin(gpt_features)               │
│ L₂ = w₁·FL(logits₁, code₁)              │ offsets = MLP_offset(gpt_features)           │
│    + w₂·FL(logits₂, code₂)              │ loss = focal(logits, GT_codes)               │
│    + w₃·‖a - â‖₁                        │      + 10000 * l1(pred_action, GT_action)    │
│                                         │                                              │
│ Focal Loss (γ=2):                       │ FL = -mean((1-p_t)^2 * log(p_t))             │
│ FL = -(1-p_t)^γ · log(p_t)              │                                              │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

**Training pseudo-code (Phase 2):**
```python
def forward(batch):
    if not vqvae_trained:  # Phase 1
        z_q, codes, vq_loss = RVQ(encoder(flatten(actions)))
        return l1(actions, decoder(z_q)) + 5 * vq_loss, {}

    # Phase 2: GPT + action head
    tokens = interleave(rgb_proj(img_feat), state_proj(state), action_queries)
    features = GPT(tokens)
    bin_logits = MLP_bin(features)       # (B, 2, codebook_size)
    offsets = MLP_offset(features)       # (B, 2, codebook_size, chunk*action_dim)
    GT_codes = frozen_RVQ.encode(GT_actions)
    pred_action = RVQ.decode(sample(bin_logits)) + offsets[sampled_codes]
    loss = focal_loss(bin_logits, GT_codes) + 10000 * l1(GT_actions, pred_action)
    return loss, {}
```

---

### 4. TD-MPC (Temporal Difference Learning for Model Predictive Control)

**Paper**: "Temporal Difference Learning for Model Predictive Control" — Hansen et al., 2022 (arXiv:2203.04955) + FOWM extension (2023, arXiv:2310.16029)
https://arxiv.org/abs/2203.04955

https://arxiv.org/pdf/2310.16029

**Category**: Offline RL / Model-Based RL with MPC Planning

**Architecture**: TOLD model (encoder + latent dynamics + reward + policy + Q-ensemble + V function)

**Training Objective**: Weighted sum of 5 losses (consistency, reward, Q, V-expectile, policy-AWR)

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Consistency:                            │ loss_c = MSE(z_pred[t+1],                    │
│ L_c = ‖ẑ_{t+1} - sg(h_φ(o_{t+1}))‖²     │            stopgrad(encode_target(obs[t+1])))│
│                                         │                                              │
│ Q-target (TD):                          │ q_target = r + γ * V(encode(next_obs))       │
│ y = r + γ · V(z')                       │                                              │
│                                         │                                              │
│ V-loss (expectile, τ=0.9):              │ diff = v_target - V(z)                       │
│ L_V = |τ - 1(d<0)| · d²                 │ loss_v = where(diff>0, τ, 1-τ) * diff²       │
│                                         │                                              │
│ Policy (AWR):                           │ adv = min(Q_target(z,a)) - V(z)              │
│ w = clamp(exp(3·A), max=100)            │ w = clamp(exp(3 * adv), max=100)             │
│ L_π = w · ‖π(z) - a‖²                   │ loss_pi = w * MSE(pi(z), actions)            │
│                                         │                                              │
│ Total: 20·L_c + 0.5·L_r + 0.1·L_Q       │ loss = 20*Lc + 0.5*Lr + 0.1*Lq               │
│       + 0.1·L_V + 0.5·L_π               │       + 0.1*Lv + 0.5*Lpi                     │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

**Inference pseudo-code (MPPI/CEM planning):**
```python
def select_action(obs):
    z = encode(obs)
    if not use_mpc: return policy(z)

    # CEM iterations
    mean, std = zeros(horizon, action_dim), max_std
    for _ in range(6):
        gaussian_actions = clamp(mean + std * randn(N), -1, 1)
        pi_actions = [policy(rollout(z)) for _ in range(N_pi)]
        all_actions = cat(gaussian_actions, pi_actions)
        values = estimate_value(z, all_actions)  # model rollout + Q
        elites = topk(values, n=50)
        score = softmax(0.5 * elite_values)
        mean = 0.1*mean + 0.9*weighted_mean(elites, score)
        std = clamp(weighted_std(elites, score), min_std, max_std)
    return sample_from_elites(score)
```

---

### 5. Gaussian Actor (SAC — Soft Actor-Critic)

**Paper**: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" — Haarnoja et al., 2018 (ICML)

https://arxiv.org/pdf/1801.01290

**Category**: Online Off-Policy RL (Maximum Entropy)

**Architecture**: CNN/MLP encoder → Gaussian policy (TanhNormal) + Q-ensemble + learnable temperature

**Training Objective**: TD error (critic) + entropy-regularized policy gradient (actor) + temperature tuning

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ TD target:                              │ y = r + γ*(1-d)*(min_Q(s',a') - α*logπ(a'))  │
│ y = r + γ(1-d)[min_j Q_j(s',a')         │                                              │
│              - α log π(a'|s')]          │                                              │
│                                         │                                              │
│ Critic loss:                            │ loss_q = Σ_j MSE(Q_j(s,a), y)                │
│ L_Q = Σ_j ‖Q_j(s,a) - y‖²               │                                              │
│                                         │                                              │
│ Actor loss:                             │ loss_pi = mean(α*log_pi - min_Q(s, a_pi))    │
│ L_π = E[α log π(a|s) - min Q(s,a)]      │                                              │
│                                         │                                              │
│ Temperature:                            │ loss_α = mean(-α * (log_pi + H_target))      │
│ L_α = -α · (log π + H_target)           │                                              │
│                                         │                                              │
│ Tanh squash log-prob:                   │ log_pi = logN(u;μ,σ)                         │
│ log π(a) = log N(u) - Σ log(1-tanh²u)   │          - sum(log(1 - tanh(u)²))            │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

**Training pseudo-code:**
```python
def update(batch):
    # Critic
    with no_grad:
        a_next, log_pi_next = actor(next_obs)
        q_target = reward + gamma * (1-done) * (min(Q_target(next_obs, a_next)) - alpha * log_pi_next)
    loss_critic = sum(MSE(Q_j(obs, action), q_target) for j in ensemble)

    # Actor
    a_pi, log_pi = actor(obs)
    loss_actor = mean(alpha * log_pi - min(Q(obs, a_pi)))

    # Temperature
    loss_alpha = mean(-alpha * (log_pi.detach() + target_entropy))

    # Target EMA
    polyak_update(Q_target, Q, tau=0.005)
```

---

### 6. PI0 (π₀)

**Paper**: "pi0: A Vision-Language-Action Flow Model for General Robot Control" — Physical Intelligence, 2024

https://arxiv.org/pdf/2410.24164

**Category**: Imitation Learning (Flow Matching VLA)

**Architecture**: PaliGemma VLM (SigLIP + Gemma 2B) + Gemma Action Expert (300M), joint attention

**Training Objective**: MSE flow-matching velocity loss

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Flow interpolation:                     │ x_t = t * noise + (1-t) * actions            │
│ x_t = t·ε + (1-t)·a                     │                                              │
│                                         │                                              │
│ Target velocity:                        │ u_t = noise - actions                        │
│ u_t = ε - a                             │                                              │
│                                         │                                              │
│ Loss:                                   │ loss = F.mse_loss(v_t, u_t)                  │
│ L = ‖v_θ(x_t,t,c) - u_t‖²               │                                              │
│                                         │                                              │
│ Time sampling:                          │ t ~ Beta(1.5, 1.0) * 0.999 + 0.001           │
│ t ~ Beta(1.5, 1.0)                      │                                              │
│                                         │                                              │
│ Euler integration:                      │ dt = -1/num_steps                            │
│ x_{t+dt} = x_t + dt · v_θ(x_t,t,c)      │ x_t = x_t + dt * v_t                         │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

**Training pseudo-code:**
```python
def forward(batch):
    noise = torch.randn_like(actions)
    t = Beta(1.5, 1.0).sample((B,)) * 0.999 + 0.001
    x_t = t * noise + (1-t) * actions
    u_t = noise - actions

    prefix_embs = concat(siglip(images), embed(lang_tokens))
    suffix_embs = MLP(concat(action_in_proj(x_t), sinusoidal(t)))

    # Joint attention through paired VLM + expert layers
    for vlm_layer, expert_layer in zip(paligemma, expert):
        prefix_embs, suffix_embs = joint_attention(prefix_embs, suffix_embs)

    v_t = action_out_proj(suffix_embs[:, -chunk_size:])
    loss = F.mse_loss(v_t, u_t)
    return loss, {}
```

**Inference pseudo-code:**
```python
def select_action(batch):
    prefix_kv = vlm_forward(images, language, use_cache=True)
    x_t = torch.randn(B, chunk_size, action_dim)
    dt = -1.0 / num_steps
    for step in range(num_steps):   # 10 steps
        t = 1.0 + step * dt
        v_t = expert_forward(state, x_t, t, prefix_kv)
        x_t = x_t + dt * v_t
    return x_t
```

---

### 7. PI0-FAST

**Paper**: "FAST: Efficient Action Tokenization for Vision-Language-Action Models" — Physical Intelligence (Pertsch et al.), 2025

https://arxiv.org/pdf/2501.09747

**Category**: Imitation Learning (Autoregressive Token Prediction VLA)

**Architecture**: PaliGemma (SigLIP + Gemma) with autoregressive action token prediction via DCT+BPE

**Training Objective**: Cross-entropy on next action token

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ DCT encoding:                           │ dct_coeffs = DCT(actions, axis=time)         │
│ C = DCT(a, axis=0, norm="ortho")        │ tokens = BPE_encode(dct_coeffs * scale)      │
│ tokens = BPE(scale · C)                 │ ids = vocab_size - 1 - skip - tokens         │
│                                         │                                              │
│ Loss:                                   │ logits = lm_head(hidden[:, -(T):-1])         │
│ L = CE(logits_{t-1}, token_t)           │ loss = CE(logits, targets[:, 1:])            │
│                                         │                                              │
│ State discretization:                   │ bins = linspace(-1, 1, 257)[:-1]             │
│ bins = linspace(-1, 1, 256)             │ discrete = digitize(state, bins) - 1         │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

**Training pseudo-code:**
```python
def forward(batch):
    input_embs = concat(siglip(images), embed(lang_tokens), embed(action_tokens))
    hidden = gemma(input_embs, custom_attention_mask)
    logits = lm_head(hidden[:, -T_action:])[:, :-1]
    targets = action_tokens[:, 1:]
    loss = masked_cross_entropy(logits, targets)
    return loss, {}
```

**Inference pseudo-code:**
```python
def predict_actions(batch):
    prefix = embed_prefix(images, language + [BOS])
    generated = autoregressive_decode(prefix, max_steps=256, temperature=T)
    action_ids = vocab_size - 1 - skip_tokens - generated
    dct_coeffs = BPE_decode(action_ids) / scale
    actions = IDCT(dct_coeffs, axis=time)
    return actions
```

---

### 8. PI0.5

**Paper**: "pi0.5: a Vision-Language-Action Model with Open-World Generalization" — Physical Intelligence, 2025

https://arxiv.org/pdf/2504.16054

**Category**: Imitation Learning (Flow Matching VLA with AdaRMS conditioning)

**Architecture**: Same as PI0 but with AdaRMS (adaptive RMS normalization) in the action expert, conditioned on timestep

**Training Objective**: MSE flow-matching velocity loss (identical formula to PI0)

Same formulas as PI0. Key difference: expert uses **AdaRMS normalization** conditioned on timestep embedding instead of plain layer norm. Also supports **RTC** (Real-Time Chunking) at inference.

---

### 9. SmolVLA

**Paper**: SmolVLA — HuggingFace, 2025 (arXiv:2506.01844)

https://arxiv.org/pdf/2506.01844

**Category**: Imitation Learning (Flow Matching with small VLA)

**Architecture**: SmolVLM2-500M (SigLIP + Gemma-like) + separate Action Expert with cross-attention

**Training Objective**: MSE flow-matching velocity loss (same formula as PI0)

Same flow matching formulas as PI0. Key differences: smaller backbone (500M), expert uses **cross-attention** (interleaved with self-attention every 2 layers) to the VLM's KV cache rather than joint concatenated attention.

---

### 10. EO1

**Paper**: PI0 architecture reimplemented on Qwen2.5-VL backbone (2024-2025)

**Category**: Imitation Learning (Flow Matching VLA)

**Architecture**: Qwen2.5-VL (3B) backbone + flow-matching action tokens injected via placeholder tokens

**Training Objective**: MSE flow-matching velocity loss (same formula as PI0)

Same core formulas as PI0. Distinguishing features: uses Qwen2.5-VL instead of PaliGemma, injects actions by replacing `<|action_pad|>` special tokens in the embedding sequence.

---

### 11. XVLA

**Paper**: XVLA — 2toINF / HuggingFace, 2025

https://arxiv.org/pdf/2510.10274

**Category**: Imitation Learning (Rectified Flow VLA)

**Architecture**: Florence-2 (BART encoder only) + deep Transformer policy head (24L, 1024d) with domain-aware soft prompts

**Training Objective**: Domain-specific losses (MSE + BCE for ee6d mode)

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Noising:                                │ x_t = noise * t + action * (1-t)             │
│ x_t = ε·t + a·(1-t)                     │                                              │
│                                         │                                              │
│ Model predicts clean action:            │ action = model(x_t, t, context)              │
│ â = f_θ(x_t, t, c)                      │                                              │
│                                         │                                              │
│ Inference (iterative refinement):       │ for i in [N..1]:                             │
│ x_t = ε·(i/N) + â·(1-i/N)               │   t = i/N                                    │
│ â = f_θ(x_t, t, c)                      │   x_t = noise*t + action*(1-t)               │
│                                         │   action = model(x_t, t, context)            │
│                                         │                                              │
│ Loss (ee6d mode):                       │ loss = 500*MSE(pos) + 10*MSE(rot)            │
│ L = 500·‖pos‖² + 10·‖rot‖²              │      + 1.0*BCE(gripper)                      │
│   + BCE(gripper)                        │                                              │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

### 12. MultiTaskDiT

**Paper**: "Large Behavior Models" (Boston Dynamics Atlas) — Jones, 2025 (arXiv:2507.05331)

https://arxiv.org/pdf/2507.05331

**Category**: Imitation Learning (Language-conditioned Diffusion/Flow DiT)

**Architecture**: DiT Transformer with AdaLN-Zero + RoPE + CLIP vision/text encoders

**Training Objective**: MSE (supports both DDPM diffusion and flow matching objectives)

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Diffusion mode:                         │ x_t = scheduler.add_noise(actions, eps, t)   │
│ x_t = √ᾱ_t·a + √(1-ᾱ_t)·ε               │ pred = DiT(x_t, t, cond)                     │
│ target = ε  (or a)                      │ loss = MSE(pred, target)                     │
│                                         │                                              │
│ Flow matching mode:                     │ x_t = t*a + (1-(1-σ_min)*t)*ε                │
│ x_t = t·a + (1-(1-σ_min)·t)·ε           │ v_target = a - (1-σ_min)*ε                   │
│ v = a - (1-σ_min)·ε                     │ loss = MSE(v_pred, v_target)                 │
│                                         │                                              │
│ AdaLN-Zero:                             │ out = x * (1 + scale) + shift                │
│ modulate(x, s, b) = x·(1+s) + b         │ where [shift, scale] = Linear(cond)          │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

### 13. Wall-X

**Paper**: "Igniting VLMs Toward the Embodied Space" — X-Square Robot, 2025 (arXiv:2509.11766)

https://arxiv.org/pdf/2509.11766

**Category**: Imitation Learning (VLA with MoE + Flow Matching + optional FAST)

**Architecture**: Qwen2.5-VL with Mixture of Experts (MoE) + ActionHead (flow matching) + cross-embodiment DOF masks

**Training Objective**: Cross-entropy (language) + flow matching MSE (actions)

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Flow matching (note: t goes 0→1):       │ noisy_a = (1-t)*noise + t*action_gt          │
│ x_t = (1-t)·ε + t·a                     │ flow_target = action_gt - noise              │
│ target = a - ε                          │                                              │
│                                         │                                              │
│ Loss:                                   │ total = CE(logits, labels)                   │
│ L = CE_lang + w · MSE_flow · dof_mask   │       + w * MSE(pred, flow_target) * mask    │
│                                         │                                              │
│ ODE integration:                        │ for t in linspace(0, 1, N):                  │
│ x_{t+dt} = x_t + dt · v_θ(x_t,t)        │   v = model(x_t, t, context)                 │
│                                         │   x_t = x_t + dt * v                         │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

### 14. Groot (GR00T-N1.5)

**Paper**: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots" — NVIDIA, 2025

https://arxiv.org/pdf/2503.14734

**Category**: Imitation Learning (Multi-embodiment Flow Matching DiT)

**Architecture**: Eagle2 VLM backbone (InternViT + LLM) + Flow-matching DiT action head with cross-attention + category-specific encoders/decoders

**Training Objective**: MSE velocity loss (flow matching)

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Flow matching:                          │ noisy_traj = (1-t)*noise + t*action          │
│ x_t = (1-t)·ε + t·a                     │ velocity_target = action - noise             │
│ v_target = a - ε                        │                                              │
│                                         │                                              │
│ t sampling:                             │ t ~ Beta(1.5, 1.0)                           │
│ t ~ Beta(1.5, 1.0), t = (s-t)/s         │ t = (0.999 - t) / 0.999                      │
│                                         │                                              │
│ DiT AdaLN:                              │ shift, scale = Linear(SiLU(temb)).chunk(2)   │
│ h = LN(x)·(1+scale) + shift             │ h = LayerNorm(x) * (1+scale) + shift         │
│                                         │                                              │
│ Euler step:                             │ actions += (1/N) * pred_velocity             │
│ a_{i+1} = a_i + (1/N)·v_θ               │                                              │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

### 15. MolmoAct2

**Paper**: MolmoAct2 — Allen AI + HuggingFace, 2026

https://arxiv.org/pdf/2605.02881v1

**Category**: Imitation Learning (Dual-mode: Flow Matching + FAST discrete tokens)

**Architecture**: Molmo VLM (27L ViT + 48L Transformer) + ActionExpert (32L DiT with per-layer cross-attention to LLM hidden states)

**Training Objective**: MSE flow loss (continuous) or CE (discrete) or both

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Flow matching:                          │ x_t = (1-t)*noise + t*actions                │
│ x_t = (1-t)·ε + t·a                     │ target_v = actions - noise                   │
│ v_target = a - ε                        │                                              │
│                                         │                                              │
│ Time:                                   │ t ~ Beta(1.0, 1.5) * scale + offset          │
│ t ~ Beta(α=1.0, β=1.5)                  │                                              │
│                                         │                                              │
│ DiT modulation (9 chunks):              │ shift, scale, gate = condition.chunk(9)      │
│ x_mod = x·(1+scale) + shift             │ output = gate * layer(x_mod)                 │
│ out = gate · layer(x_mod)               │                                              │
│                                         │                                              │
│ Knowledge insulation:                   │ k, v = k.detach(), v.detach()                │
│ Cross-attn KV from LLM: detached        │ # prevents LLM drift from action loss        │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

### 16. VLA-JEPA

**Paper**: "VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model" — Sun et al., 2026 (arXiv:2602.10098)

https://arxiv.org/pdf/2602.10098

**Category**: Imitation Learning (Flow Matching + World Model Auxiliary)

**Architecture**: Qwen3-VL (2B) + DiT-B action head (16L, 768d) + V-JEPA2 world model (frozen ViT-L encoder + trainable predictor)

**Training Objective**: MSE velocity (action) + L1 latent prediction (world model)

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Action loss:                            │ x_t = (1-t)*noise + t*actions                │
│ L_action = MSE(v_θ, a-ε)                │ loss_a = MSE(DiT(x_t,t,ctx), actions-noise)  │
│                                         │                                              │
│ World model loss:                       │ pred = predictor(video_emb[:ctx], act_tok)   │
│ L_wm = ‖predictor(z_{<t}, a) - z_t‖₁    │ loss_w = L1(pred, video_emb[1:])             │
│                                         │                                              │
│ Total:                                  │ loss = loss_a + 0.1 * loss_w                 │
│ L = L_action + 0.1 · L_wm               │                                              │
│                                         │                                              │
│ Repeated diffusion (variance red.):     │ actions = actions.repeat(8, 1, 1)            │
│ Each sample replicated 8x               │ # independent noise per replica              │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

### 17. RTC (Real-Time Chunking)

**Paper**: "Real-Time Execution of Action Chunking Flow Policies" — Black, Galliker, Levine, 2025 (arXiv:2506.07339)

https://arxiv.org/pdf/2506.07339

**Category**: Inference-time enhancement (not a standalone policy)

**Architecture**: Wraps any flow-matching policy (PI0, PI0.5, SmolVLA) with guidance at denoising time

**Purpose**: Maintains temporal consistency between consecutive overlapping action chunks

#### Formulas + Pseudo-code

```
┌─────────────────────────────────────────┬──────────────────────────────────────────────┐
│           FORMULA                       │              CODE MAPPING                    │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Predicted clean action:                 │ x1_t = x_t - time * v_t                      │
│ x̂₀ = x_t - t · v_t                      │                                              │
│                                         │                                              │
│ Guidance error:                         │ err = (prev_chunk - x1_t) * weights          │
│ e = (a_prev - x̂₀) · w                   │                                              │
│                                         │                                              │
│ Correction gradient:                    │ correction = autograd.grad(x1_t, x_t,        │
│ ∇ = ∂x̂₀/∂x_t · e                        │              grad_outputs=err)               │
│                                         │                                              │
│ Guidance weight (adaptive):             │ τ = 1 - time                                 │
│ c = (1-τ)/τ · inv_r²                    │ c = ((1-τ)/τ) * (sq_omt + τ²)/sq_omt         │
│                                         │ c = clamp(c, max=max_weight)                 │
│                                         │                                              │
│ Guided velocity:                        │ v_guided = v_t - c * correction              │
│ v_guided = v_t - c · ∇                  │                                              │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## How to Create a New Policy

### Step-by-step checklist:

1. **Create directory**: `src/lerobot/policies/my_policy/`

2. **Configuration** (`configuration_my_policy.py`):
```python
from dataclasses import dataclass, field
from lerobot.configs import PreTrainedConfig

@PreTrainedConfig.register_subclass("my_policy")
@dataclass
class MyPolicyConfig(PreTrainedConfig):
    # Your hyperparameters here
    hidden_dim: int = 256
    chunk_size: int = 10
    n_action_steps: int = 5
```

3. **Modeling** (`modeling_my_policy.py`):
```python
from lerobot.policies.pretrained import PreTrainedPolicy

class MyPolicy(PreTrainedPolicy):
    config_class = MyPolicyConfig
    name = "my_policy"

    def __init__(self, config: MyPolicyConfig, **kwargs):
        super().__init__(config)
        # Build your neural network here

    def get_optim_params(self) -> dict:
        return {"all": {"params": self.parameters()}}

    def reset(self):
        self._action_queue = []

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        # Training: compute loss from batch
        loss = ...
        return loss, {"info_key": info_value}

    def predict_action_chunk(self, batch, **kwargs) -> Tensor:
        # Full chunk prediction (no queue logic)
        return action_chunk  # (B, chunk_size, action_dim)

    def select_action(self, batch, **kwargs) -> Tensor:
        # Manages queue, calls predict_action_chunk when needed
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)
            self._action_queue = list(chunk.split(1, dim=1))
        return self._action_queue.pop(0).squeeze(1)
```

4. **Processor** (`processor_my_policy.py`):
```python
def make_my_policy_pre_post_processors(config, dataset_stats=None):
    preprocessor = PolicyProcessorPipeline(steps=[...])
    postprocessor = PolicyProcessorPipeline(steps=[...])
    return preprocessor, postprocessor
```

5. **Register in factory** (`factory.py`):
   - Add config import at top
   - Add `elif name == "my_policy"` in `get_policy_class()`
   - Add `elif policy_type == "my_policy"` in `make_policy_config()`
   - Add `elif isinstance(policy_cfg, MyPolicyConfig)` in `make_pre_post_processors()`

6. **Run**: `lerobot-train --policy.type=my_policy ...`

### Design Pattern Summary

```
PreTrainedPolicy (abstract base)
├── config_class (class attribute) → your config dataclass
├── name (class attribute) → string for factory lookup
├── __init__(config) → build all nn.Modules
├── get_optim_params() → parameter groups for optimizer
├── reset() → clear state between episodes
├── forward(batch) → (loss, info) for training
├── predict_action_chunk(batch) → raw chunk prediction
└── select_action(batch) → single action with queue management
```

The training script (`scripts/train.py`) calls `forward()` in a loop. The eval script calls `select_action()` in a loop with `reset()` between episodes. The preprocessor/postprocessor handle normalization transparently.
