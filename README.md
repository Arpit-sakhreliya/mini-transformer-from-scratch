# Tiny LLM from Scratch

A minimal decoder-only transformer trained entirely in NumPy — no PyTorch, no frameworks.
Built to learn how language models actually work under the hood: embeddings, attention, MLP,
layer norm, backprop, and autoregressive generation, all implemented by hand.

---

## Table of Contents

1. [What This Model Is](#what-this-model-is)
2. [Architecture Overview](#architecture-overview)
3. [Layer-by-Layer Walkthrough](#layer-by-layer-walkthrough)
   - [Layer 0 — Vocabulary and Tokenization](#layer-0--vocabulary-and-tokenization)
   - [Layer 1 — Token Embeddings](#layer-1--token-embeddings)
   - [Layer 2 — Self-Attention](#layer-2--self-attention)
   - [Layer 3 — Layer Normalization](#layer-3--layer-normalization)
   - [Layer 4 — MLP (Feed-Forward Block)](#layer-4--mlp-feed-forward-block)
   - [Layer 5 — Output Projection](#layer-5--output-projection)
   - [Layer 6 — Softmax and Loss](#layer-6--softmax-and-loss)
4. [Two-Block Stack](#two-block-stack)
5. [Training: Forward → Loss → Backward → Update](#training-forward--loss--backward--update)
   - [Forward Pass](#forward-pass)
   - [Cross-Entropy Loss](#cross-entropy-loss)
   - [Backpropagation](#backpropagation)
   - [Gradient Descent Update](#gradient-descent-update)
6. [Autoregressive Generation](#autoregressive-generation)
7. [Model Dimensions at a Glance](#model-dimensions-at-a-glance)
8. [What the Model Learns](#what-the-model-learns)
9. [How to Run](#how-to-run)
10. [Limitations and What to Try Next](#limitations-and-what-to-try-next)

---

## What This Model Is

This is a **decoder-only transformer** — the same fundamental architecture behind GPT-style
language models — shrunk down to the smallest possible size so every computation is visible
and understandable.

It learns to predict the next token in a sequence. Given `["I", " ", "like"]`, it should
predict `" "`. Given `["I", " ", "like", " "]`, it should predict `"cats"` or `"dogs"`.
After enough training it can complete entire sentences autoregressively from a single starting word.

**Model size in this code:**

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 10 tokens |
| Embedding dimension `d` | 8 |
| MLP hidden dimension | 16 |
| Transformer blocks | 2 |
| Total weight matrices | 4×attn + 2×MLP per block + 1 output = 13 matrices |

---

## Architecture Overview

```
Input tokens (list of strings)
        │
        ▼
┌───────────────────┐
│   Token Embedding │   vocab_id → 8-dim vector
└────────┬──────────┘
         │  X  (seq_len, 8)
         ▼
┌─────────────────────────────────────┐
│           TRANSFORMER BLOCK 1       │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  Self-Attention              │   │
│  │  WQ1, WK1, WV1, WO1         │   │
│  └──────────────┬───────────────┘   │
│                 │ A1                │
│  ┌──────────────▼───────────────┐   │
│  │  Layer Norm                  │   │
│  └──────────────┬───────────────┘   │
│                 │ A1_ln             │
│  ┌──────────────▼───────────────┐   │
│  │  MLP   W1_1 (8→16)           │   │
│  │        ReLU                  │   │
│  │        W2_1 (16→8)           │   │
│  └──────────────┬───────────────┘   │
│                 │ H1                │
│  ┌──────────────▼───────────────┐   │
│  │  Layer Norm                  │   │
│  └──────────────┬───────────────┘   │
│                 │ H1_ln             │
└─────────────────┼───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│           TRANSFORMER BLOCK 2       │
│       (same structure, own weights) │
│                 │ H2_ln             │
└─────────────────┼───────────────────┘
                  │
         ┌────────▼────────┐
         │  Output Layer   │   W_out: (8, 10)
         └────────┬────────┘
                  │  logits  (seq_len, 10)
         ┌────────▼────────┐
         │    Softmax      │
         └────────┬────────┘
                  │  probs over vocabulary
         ┌────────▼────────┐
         │  Cross-Entropy  │
         │      Loss       │
         └─────────────────┘
```

---

## Layer-by-Layer Walkthrough

### Layer 0 — Vocabulary and Tokenization

Before anything numerical happens, raw text must be converted to integers. This model uses a
hand-written vocabulary of 10 tokens:

```
{ "I":0, "like":1, "cats":2, "dogs":3, " ":4,
  "you":5, "hate":6, "birds":7, ".":8, "<END>":9 }
```

Each token maps to a unique integer ID. The special token `<END>` signals the model to stop
generating. Real LLMs use tokenizers (BPE, WordPiece) that can handle any text; here the
vocabulary is fixed and tiny so we can focus on the architecture.

A sentence like `["I", " ", "like", " ", "cats"]` becomes the index sequence `[0, 4, 1, 4, 2]`.

---

### Layer 1 — Token Embeddings

**Concept:** A raw token ID is just a number — it has no meaning by itself. The embedding layer
converts each integer into a dense vector of real numbers that the model can do math on.
Semantically similar tokens should eventually land near each other in this space.

**Mathematically:**

The embedding matrix `E` has shape `(vocab_size, d) = (10, 8)`. Each row is a learnable
vector for one token. Looking up token `i` is just:

```
x_i = E[i]     →  shape: (8,)
```

For a sequence of `n` tokens, stacking all lookups gives:

```
X = E[token_ids]     →  shape: (n, 8)
```

Each row of `X` is the embedding of one token. Position is implicitly encoded by row order
(real transformers add explicit positional encodings; this model omits them for simplicity).

**In code:**
```python
embedding_matrix = np.random.randn(vocab_size, embedding_dim)   # (10, 8)

def embed(sentence):
    return np.array([embedding_matrix[vocab[word]] for word in sentence])
```

**What is learned:** The embedding matrix is updated during backpropagation. Over training,
the model nudges the vector for `"cats"` closer to `"dogs"` (both follow `"like"` or `"hate"`)
and keeps `"I"` and `"you"` distinct (they start different sentences).

---

### Layer 2 — Self-Attention

Self-attention is the core of the transformer. It lets every token in the sequence look at
every other token and decide how much to borrow from each.

**Intuition:** Suppose the model sees `["I", " ", "like", " ", "cats"]`. When computing the
representation of `"cats"`, attention lets it ask: *"which earlier tokens are relevant to
understanding me?"* It will learn to look back at `"like"` heavily, because knowing the verb
helps predict what comes next.

**The three projections — Q, K, V:**

Each token's embedding is linearly projected three times:

```
Q = X @ WQ     (queries:  "what am I looking for?")
K = X @ WK     (keys:     "what do I contain?")
V = X @ WV     (values:   "what do I send if I'm attended to?")
```

All three have shape `(seq_len, d)`. `WQ`, `WK`, `WV` are learned weight matrices of shape
`(d, d) = (8, 8)`.

**Attention scores:**

How much should token `i` attend to token `j`? Compute the dot product of query `i` with key `j`:

```
scores[i, j] = Q[i] · K[j]
```

Arranged into a matrix:

```
scores = Q @ K^T     →  shape: (seq_len, seq_len)
```

The `sqrt(d)` scaling keeps dot products from growing too large as `d` increases (otherwise
softmax saturates):

```
scores = (Q @ K^T) / sqrt(d)
```

**Softmax into attention weights:**

Convert scores to probabilities row-wise so each token's attention weights sum to 1:

```
weights[i, :] = softmax(scores[i, :])
```

`weights[i, j]` is the fraction of token `j`'s information that token `i` will incorporate.

**Weighted aggregation:**

Each token's new representation is a weighted sum of all value vectors:

```
output_pre_WO = weights @ V     →  shape: (seq_len, d)
```

**Output projection:**

A final linear transformation mixes the attended information across the `d` dimensions:

```
output = output_pre_WO @ WO
```

`WO` is a `(d, d)` learned matrix. This lets the model recombine the aggregated values in a
task-specific way.

**Full formula:**

```
Attention(X) = softmax( (X WQ)(X WK)^T / sqrt(d) ) · (X WV) · WO
```

**Numerical stability:** The code subtracts `max(scores)` before the exponential to prevent
overflow — this is a standard trick that doesn't change the softmax output:

```python
weights = exp(scores - max(scores)) / sum(exp(scores - max(scores)))
```

**What is learned:** `WQ`, `WK`, `WV`, `WO` are all trained. The model learns which query/key
pairings to make strong (which tokens to look at) and how to mix the resulting values.

---

### Layer 3 — Layer Normalization

**Concept:** As tensors flow through attention and MLP layers, their scale can drift — some
dimensions grow large, others shrink. Layer norm re-centers and rescales each token's vector
independently to have mean ≈ 0 and std ≈ 1. This stabilizes training, especially across
multiple stacked blocks.

**Mathematics:**

For a vector `x` of length `d`:

```
μ = mean(x)
σ = std(x)
x̂ = (x - μ) / (σ + ε)
```

`ε = 1e-5` prevents division by zero. The result `x̂` has mean 0 and std 1.

Real transformer implementations include learnable gain `γ` and bias `β` per dimension:

```
LayerNorm(x) = γ ⊙ x̂ + β
```

This model omits `γ` and `β` to keep things minimal (equivalent to fixing both to 1 and 0).

**In code:**
```python
def layernorm(x, eps=1e-5):
    return (x - x.mean()) / (x.std() + eps)
```

Note: this normalizes over the entire matrix `x` (all tokens and all dimensions together),
not per-token. A production implementation would normalize each token's vector independently.
For this toy scale it has the same stabilizing effect.

**In backpropagation**, the gradient through layer norm involves the Jacobian of the
normalization operation. The simplified formula used here is:

```
dL/dx = (1/σ) * ( dL/dx̂  -  mean(dL/dx̂)  -  x̂ · mean(dL/dx̂ · x̂) )
```

This correctly propagates gradients while accounting for the mean-subtraction and
std-division in the forward pass.

---

### Layer 4 — MLP (Feed-Forward Block)

**Concept:** The attention layer mixes information *across* tokens (token-to-token
communication). The MLP then processes each token *independently*, expanding and compressing
its representation. This is where the model stores and retrieves factual associations.

**Architecture:**

```
hidden = ReLU(X @ W1)       expand:    (seq_len, 8)  →  (seq_len, 16)
output = hidden @ W2         compress:  (seq_len, 16) →  (seq_len, 8)
```

`W1` has shape `(8, 16)` and `W2` has shape `(16, 8)`. The hidden dimension 16 is 2× the
embedding dimension — real GPT models typically use 4×.

**ReLU:**

```
ReLU(x) = max(0, x)
```

Without a non-linearity between `W1` and `W2`, the two matrix multiplications would collapse
into one and the MLP would have no more capacity than a single linear layer. ReLU introduces
non-linearity, enabling the MLP to learn arbitrary functions.

**What is learned:** `W1` and `W2` act like pattern-matching lookup tables. One row of `W1`
might activate strongly for "token following a subject word"; `W2` then converts that
activation into a useful direction in the embedding space.

**Backpropagation through MLP:**

Starting from gradient `dL/dOutput`:

```
dL/dW2      = hidden^T   @  dL/dOutput       (outer product accumulation)
dL/dhidden  = dL/dOutput @  W2^T

dL/dhidden_pre[i] = dL/dhidden[i]   if hidden_pre[i] > 0    (ReLU gate)
                  = 0                otherwise

dL/dW1      = X^T          @  dL/dhidden_pre
dL/dX       = dL/dhidden_pre @  W1^T
```

---

### Layer 5 — Output Projection

After two transformer blocks, the final hidden state `H2_ln` has shape `(seq_len, 8)`. The
output layer maps this to **logits** over the vocabulary:

```
logits = H2_ln @ W_out     →  shape: (seq_len, vocab_size) = (seq_len, 10)
```

`W_out` has shape `(8, 10)`. Each row of `logits` is a score vector over all 10 possible
next tokens, one row per position in the sequence.

**Only the last row is used for prediction.** The model predicts what comes *after* the last
token in the current input:

```python
last_logits = logits[-1]     # shape: (10,)
```

This is the decoder/causal paradigm: position `i` predicts what comes at position `i+1`.

**Backpropagation:**

```
dL/dW_out   = H2_ln[-1]^T  ⊗  dL/dlogits     (outer product, only last token)
dL/dH2_ln   = dL/dlogits   @  W_out^T         (only last row non-zero)
```

---

### Layer 6 — Softmax and Loss

**Softmax** converts raw logits into a valid probability distribution (all positive, sum to 1):

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

Numerically stable version (subtract max before exp):

```
p_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
```

**Cross-Entropy Loss** measures how wrong the predicted distribution is compared to the true
target. If the correct next token has index `t`:

```
L = -log(p_t)
```

This is maximized (worst) when `p_t = 0` and minimized (best) when `p_t = 1`. For a perfect
prediction, `L = -log(1) = 0`.

Over a full training corpus of `N` pairs, the total loss sums across all predictions:

```
L_total = Σ_i  -log(p_{t_i})
```

**The key gradient — softmax + cross-entropy combined:**

The gradient of `L` with respect to the logit vector `z` has the elegant form:

```
dL/dz_i = p_i - 1{i == t}
```

Which in code is:

```python
grad_logits = probs.copy()
grad_logits[target_id] -= 1
```

Subtract 1 from the correct class. This gradient is small when the model is already confident
about the right answer and large when it's confidently wrong.

---

## Two-Block Stack

This model stacks two transformer blocks. Each block is:

```
Attention → LayerNorm → MLP → LayerNorm
```

Each block has its own independent weight matrices (`WQ1/WK1/...` for block 1 and
`WQ2/WK2/...` for block 2). They are initialized with different random seeds.

**Why stack blocks?**

One block of attention + MLP can only represent one level of abstraction. With two blocks:

- **Block 1** might learn low-level associations: which words tend to follow each other.
- **Block 2** can build higher-level patterns on top: subject-verb-object relationships across the sentence.

Real LLMs stack dozens to hundreds of blocks (GPT-3 has 96).

**Data flow through both blocks:**

```
X           →  Attention1  →  A1
A1          →  LayerNorm   →  A1_ln
A1_ln       →  MLP1        →  H1
H1          →  LayerNorm   →  H1_ln

H1_ln       →  Attention2  →  A2
A2          →  LayerNorm   →  A2_ln
A2_ln       →  MLP2        →  H2
H2          →  LayerNorm   →  H2_ln

H2_ln       →  W_out       →  logits
```

Note: this architecture does **not** include residual connections (`x + sublayer(x)` shortcuts
present in real transformers). Residuals help gradients flow through deep networks; their
absence here is acceptable at 2 blocks but would cause vanishing gradients at depth.

---

## Training: Forward → Loss → Backward → Update

### Forward Pass

For each training pair `(input_seq, target_token)`:

1. Embed `input_seq` → `X`
2. Run through Block 1 (Attention → LayerNorm → MLP → LayerNorm) → `H1_ln`
3. Run through Block 2 → `H2_ln`
4. Project to logits → `logits`
5. Extract `logits[-1]` — the prediction for the next token
6. Compute cross-entropy loss against `vocab[target_token]`

### Cross-Entropy Loss

```
L = -log( softmax(logits[-1])[target_id] )
```

The total loss across a full epoch sums over every `(input, target)` pair in every sentence:

```
L_epoch = Σ_{sentences} Σ_{positions} -log(p_{correct at that position})
```

In the training loop, every prefix is a training example. For the sentence
`["I", " ", "like", " ", "cats", " ", ".", "<END>"]`, the pairs are:

```
["I"]                              → " "
["I", " "]                         → "like"
["I", " ", "like"]                 → " "
["I", " ", "like", " "]            → "cats"
...
```

7 training pairs from one 8-token sentence.

### Backpropagation

Backprop applies the chain rule from the loss all the way back through every layer, computing
`dL/dW` for every weight matrix. The gradient flows in reverse order of the forward pass:

```
dL/dlogits
    │
    ▼  output layer backward
dL/dW_out,   dL/dH2_ln
    │
    ▼  layernorm backward  (through H2_ln = layernorm(H2))
dL/dH2
    │
    ▼  mlp_backward (Block 2)
dL/dW1_2,  dL/dW2_2,   dL/dA2_ln
    │
    ▼  layernorm backward  (through A2_ln = layernorm(A2))
dL/dA2
    │
    ▼  attention_backward (Block 2)
dL/dWQ2, dL/dWK2, dL/dWV2, dL/dWO2,   dL/dH1_ln
    │
    ▼  layernorm backward  (through H1_ln = layernorm(H1))
dL/dH1
    │
    ▼  mlp_backward (Block 1)
dL/dW1_1,  dL/dW2_1,   dL/dA1_ln
    │
    ▼  layernorm backward  (through A1_ln = layernorm(A1))
dL/dA1
    │
    ▼  attention_backward (Block 1)
dL/dWQ1, dL/dWK1, dL/dWV1, dL/dWO1,   dL/dX
    │
    ▼  embed_backward
embedding_matrix updated in-place
```

**Attention backward in detail:**

Given upstream gradient `dL/dOutput`:

```
1. WO:       grad_WO        = output_pre_WO^T  @  dL/dOutput
             dL/d(pre_WO)   = dL/dOutput        @  WO^T

2. V:        grad_V         = weights^T         @  dL/d(pre_WO)
             dL/dweights    = dL/d(pre_WO)      @  V^T

3. Softmax:  for each row i:
             dL/dscores[i]  = weights[i] ⊙ (dL/dweights[i] - weights[i]·dL/dweights[i])
             (Jacobian-vector product of row-wise softmax)

4. Q, K:     dL/dQ          = dL/dscores        @  K / sqrt(d)
             dL/dK          = dL/dscores^T       @  Q / sqrt(d)

5. WQ,WK,WV: grad_WQ        = X^T               @  dL/dQ
             grad_WK        = X^T               @  dL/dK
             grad_WV        = X^T               @  dL/dV

6. X:        dL/dX          = dL/dQ @ WQ^T  +  dL/dK @ WK^T  +  dL/dV @ WV^T
```

### Gradient Descent Update

After computing all gradients, each weight matrix is nudged in the direction that reduces loss:

```
W ← W - lr · dL/dW
```

With `learning_rate = 0.01`. This is vanilla SGD (stochastic gradient descent) — one update
per training pair, no momentum, no adaptive learning rates (real models use Adam).

---

## Autoregressive Generation

After training, the model generates text token-by-token. This is called **autoregressive**
generation because each new token is fed back in as part of the input for the next step.

```
Step 1:  input = ["I"]              →  predict " "
Step 2:  input = ["I", " "]         →  predict "like"
Step 3:  input = ["I", " ", "like"] →  predict " "
...
Step N:  ...                         →  predict "<END>"  →  stop
```

At each step, the model runs a full forward pass over the entire current sequence, reads only
`logits[-1]` (the prediction for the next position), applies softmax to get probabilities, and
picks the most probable token (`argmax`).

**Temperature** controls the sharpness of sampling:

```
scaled_logits = logits / temperature
probs         = softmax(scaled_logits)
```

- `temperature = 1.0` — use the raw probabilities
- `temperature < 1.0` — sharper, more confident (greedy)
- `temperature > 1.0` — flatter, more random

The `generate()` function runs this loop until `<END>` or `max_tokens`:

```python
def generate(input_seq, max_tokens=20):
    current_seq = list(input_seq)
    for _ in range(max_tokens):
        logits, _ = model_forward(current_seq)
        pred_id, _ = predict_next_token(logits)
        next_token = id_to_token[pred_id]
        current_seq.append(next_token)
        if next_token == "<END>":
            break
    return current_seq
```

The `evaluate()` function calls `generate()` on 5 randomly chosen dataset sentences,
using only the first token as the prompt, and prints the result vs. the expected output.

---

## Model Dimensions at a Glance

| Tensor | Shape | Description |
|--------|-------|-------------|
| `embedding_matrix` | (10, 8) | One 8-dim vector per vocabulary token |
| `WQ1, WK1, WV1, WO1` | (8, 8) each | Block 1 attention weight matrices |
| `WQ2, WK2, WV2, WO2` | (8, 8) each | Block 2 attention weight matrices |
| `W1_1` | (8, 16) | Block 1 MLP expand |
| `W2_1` | (16, 8) | Block 1 MLP compress |
| `W1_2` | (8, 16) | Block 2 MLP expand |
| `W2_2` | (16, 8) | Block 2 MLP compress |
| `W_out` | (8, 10) | Final projection to vocabulary logits |
| `X` | (seq_len, 8) | Token embeddings for current input |
| `Q, K, V` | (seq_len, 8) each | Attention projections |
| `weights` | (seq_len, seq_len) | Attention weight matrix |
| `logits` | (seq_len, 10) | Per-position vocabulary scores |

**Total parameter count:**

```
Embeddings:          10 × 8  =   80
Block 1 attention:   4 × 8×8 =  256
Block 1 MLP:         8×16 + 16×8 = 256
Block 2 attention:   4 × 8×8 =  256
Block 2 MLP:         8×16 + 16×8 = 256
Output:              8 × 10  =   80
─────────────────────────────────────
Total:                          1,184
```

GPT-3 has 175 billion parameters. This model has 1,184.

---

## What the Model Learns

The dataset has 5 sentences:

```
"I like cats ."
"I like dogs ."
"you hate birds ."
"you like cats ."
"I hate birds ."
```

After enough training epochs (try 200+), the model should learn:

- Sentences starting with `"I"` are followed by `" "` then `"like"` or `"hate"`
- `"like"` and `"hate"` are always followed by `" "` then an animal word
- Every sentence ends with `" . <END>"`
- `"birds"` only follows `"hate"` (never `"like"` in this dataset)
- `"cats"` and `"dogs"` only follow `"like"`

At 20 epochs the loss is still high (~45) and generation is mostly wrong. At 200 epochs it
drops significantly and the model starts completing sentences correctly. This illustrates
exactly why real LLMs require billions of parameters and trillions of training tokens.

---

## How to Run

```bash
python tiny_llm.py
```

To train longer, change the epoch count:

```python
train(epochs=200)   # much better results
```

To evaluate at any point, call:

```python
evaluate(n=5)       # n = number of random samples to show
```

To generate from a custom prompt:

```python
result = generate(["you"])
print("".join(result))
```

---

## Limitations and What to Try Next

**This model omits several things real transformers include:**

| Missing piece | What it does | Where to add |
|---------------|-------------|--------------|
| Positional encoding | Tells the model the position of each token | Add a learned or sinusoidal `pos_embedding` to `X` in `embed()` |
| Residual connections | `x + sublayer(x)` — enables deep stacking | Wrap each block: `A1 = X + attention_forward(X)`, `H1 = A1_ln + mlp_forward(A1_ln)` |
| Causal masking | Prevents token `i` from attending to token `j > i` | Apply upper-triangular `-inf` mask to `scores` before softmax |
| Multi-head attention | Multiple attention patterns in parallel | Split `Q/K/V` into `h` heads, compute attention per head, concatenate |
| Learnable LayerNorm | `γ` and `β` per dimension | Add `gamma`, `beta` arrays; multiply/add after normalization |
| Adam optimizer | Adaptive learning rates per parameter | Replace `W -= lr * dW` with momentum + adaptive scale |
| Dropout | Regularization during training | Zero random elements of attention weights or MLP hidden layer |


# Mini Transformer: End-to-End Walkthrough

A single forward pass, loss computation, and backward pass — traced with exact matrix shapes and concrete numbers for the input `["I", " ", "like"]` predicting `" "`.

---

## The Setup

```
vocab_size  = 10    embedding_dim d = 8    MLP hidden = 16    blocks = 2
```

**Vocabulary:**
```
"I"→0  " "→4  "like"→1  "cats"→2  "dogs"→3
"you"→5  "hate"→6  "birds"→7  "."→8  "<END>"→9
```

**Input sequence:** `["I", " ", "like"]`  → token IDs `[0, 4, 1]`  
**Target token:** `" "` → ID `4`

---

## FORWARD PASS

---

### Step 1 — Token Embedding

`embedding_matrix E` has shape **(10 × 8)** — one 8-dimensional row per vocabulary token.

Look up the three input IDs:

```
X = E[[0, 4, 1]]   →   shape: (3, 8)

X = [ E[0],    ← row for "I"
      E[4],    ← row for " "
      E[1] ]   ← row for "like"
```

Each row is a learned 8-dimensional vector. Row order encodes position.

---

### Step 2 — Transformer Block 1: Self-Attention

**Weights:** `WQ1, WK1, WV1, WO1` each shape **(8 × 8)**

**Project X into Q, K, V:**

```
Q = X @ WQ1    →   shape: (3, 8)      "what am I looking for?"
K = X @ WK1    →   shape: (3, 8)      "what do I contain?"
V = X @ WV1    →   shape: (3, 8)      "what do I send if attended to?"
```

**Compute attention scores** — how much should each token attend to every other?

```
scores = (Q @ Kᵀ) / √8    →   shape: (3, 3)

         "I"    " "   "like"
"I"   [ s₀₀   s₀₁   s₀₂ ]
" "   [ s₁₀   s₁₁   s₁₂ ]
"like"[ s₂₀   s₂₁   s₂₂ ]
```

Dividing by `√8 ≈ 2.83` prevents dot products from growing too large, which would saturate softmax.

**Convert scores to attention weights** (softmax row-wise):

```
weights = softmax(scores, axis=1)    →   shape: (3, 3)

weights[i, j] = probability that token i attends to token j
Each row sums to 1.
```

Example — if "like" (row 2) has learned to focus on " " (position 1):
```
weights[2] ≈ [0.05, 0.90, 0.05]
```

**Weighted sum of values:**

```
context = weights @ V         →   shape: (3, 8)
```

Each token's new representation is a mixture of all value vectors, weighted by attention.

**Output projection:**

```
A1 = context @ WO1            →   shape: (3, 8)
```

---

### Step 3 — Layer Norm (after Attention)

Normalize every element in A1 to stabilize scale:

```
μ = mean(A1)
σ = std(A1)
A1_ln = (A1 - μ) / (σ + 1e-5)    →   shape: (3, 8)
```

---

### Step 4 — Transformer Block 1: MLP

**Weights:** `W1_1` shape **(8 × 16)**, `W2_1` shape **(16 × 8)**

```
hidden_pre = A1_ln @ W1_1        →   shape: (3, 16)    [expand]
hidden     = ReLU(hidden_pre)    →   shape: (3, 16)    [gate negatives to 0]
H1         = hidden @ W2_1       →   shape: (3, 8)     [compress back]
```

**ReLU:** `max(0, x)` — keeps positive activations, zeros out negatives. Without it, two matrix multiplications collapse into one linear transformation.

---

### Step 5 — Layer Norm (after MLP)

```
H1_ln = layernorm(H1)    →   shape: (3, 8)
```

---

### Step 6 — Transformer Block 2

Identical structure to Block 1, with **its own independent weights** `WQ2, WK2, WV2, WO2, W1_2, W2_2`.

Input is `H1_ln`. Output is `H2_ln`, shape **(3, 8)**.

Block 1 learns low-level co-occurrence patterns. Block 2 builds higher-level structure on top.

---

### Step 7 — Output Projection

**Weight:** `W_out` shape **(8 × 10)**

```
logits = H2_ln @ W_out    →   shape: (3, 10)
```

Each row is a score over all 10 vocabulary tokens. We only care about the **last row** — the prediction for what comes after `"like"`:

```
last_logits = logits[-1]    →   shape: (10,)

[ score_"I", score_" ", score_"like", score_"cats", score_"dogs",
  score_"you", score_"hate", score_"birds", score_".", score_"<END>" ]
```

---

### Step 8 — Softmax → Probabilities

```
probs = softmax(last_logits)    →   shape: (10,)

probs[i] = exp(last_logits[i]) / Σⱼ exp(last_logits[j])
```

All values positive, sum to 1. Example at an untrained model:

```
probs ≈ [0.08, 0.12, 0.09, 0.11, 0.10, 0.09, 0.11, 0.10, 0.10, 0.10]
          "I"   " "  "like" "cats" "dogs" "you" "hate" "birds" "." "<END>"
```

The model predicts `" "` with probability 0.12 — barely better than random.

---

## LOSS COMPUTATION

---

### Cross-Entropy Loss

Target is `" "` → ID `4`.

```
L = -log(probs[4])
  = -log(0.12)
  ≈ 2.12
```

**Interpretation:**
| Loss | Meaning |
|------|---------|
| 0.0 | Perfect — `probs[4] = 1.0` |
| 2.30 | Random — `probs[4] = 0.10` (uniform over 10 tokens) |
| >4.0 | Confidently wrong |

At the start of training, loss ≈ 2.3 (random). After 400 epochs on this tiny dataset it converges toward 0.

---

## BACKWARD PASS

The chain rule propagates `dL/dW` from the output back through every layer. Each gradient tells each weight: *"move this much in this direction to reduce loss."*

---

### Step B1 — Gradient of Loss w.r.t. Logits

The softmax + cross-entropy gradient has a clean closed form:

```
∂L/∂last_logits[i] = probs[i] - 1{i == 4}
```

In practice: subtract 1 from the correct class slot, leave others unchanged.

```
grad_logits = [0.08,  0.12,  0.09,  0.11,  0.10, ...]
                                            ↑
                                subtract 1 here (target = 4)
            = [0.08,  0.12,  0.09,  0.11, -0.90, ...]
```

Large negative on the correct class: the model needs to push its score up hard. Small positives elsewhere: those need to come down slightly.

---

### Step B2 — Gradient through W_out

```
logits[-1] = H2_ln[-1] @ W_out
```

```
∂L/∂W_out    = H2_ln[-1]ᵀ ⊗ grad_logits    →   shape: (8, 10)
∂L/∂H2_ln    = zeros(3, 8),  except last row = grad_logits @ W_outᵀ
               →   shape: (3, 8)
```

Only the last token row contributes to the loss — the gradient is zero for positions 0 and 1.

---

### Step B3 — Gradient through Layer Norm

For `H2_ln = layernorm(H2)`:

```
∂L/∂H2 = (1/σ) · ( ∂L/∂H2_ln  −  mean(∂L/∂H2_ln)  −  H2_ln · mean(∂L/∂H2_ln · H2_ln) )
```

This accounts for the coupling introduced by mean-subtraction and std-division across the matrix.

```
→   shape: (3, 8)
```

---

### Step B4 — Gradient through MLP Block 2

Working backwards through `H2 = hidden @ W2_2`:

```
∂L/∂W2_2  = hiddenᵀ @ ∂L/∂H2          →   shape: (16, 8)
∂L/∂hidden = ∂L/∂H2 @ W2_2ᵀ           →   shape: (3, 16)
```

Through ReLU — the gate: gradient only passes where the pre-activation was positive:

```
∂L/∂hidden_pre[i,j] = ∂L/∂hidden[i,j]   if hidden_pre[i,j] > 0
                     = 0                  otherwise
```

Through `hidden_pre = A2_ln @ W1_2`:

```
∂L/∂W1_2  = A2_lnᵀ @ ∂L/∂hidden_pre    →   shape: (8, 16)
∂L/∂A2_ln = ∂L/∂hidden_pre @ W1_2ᵀ     →   shape: (3, 8)
```

---

### Step B5 — Gradient through Attention Block 2

Given upstream gradient `∂L/∂A2`:

```
1. WO:   ∂L/∂WO2          = context_preᵀ @ ∂L/∂A2       →  shape: (8, 8)
         ∂L/∂context_pre  = ∂L/∂A2 @ WO2ᵀ               →  shape: (3, 8)

2. V:    ∂L/∂V            = weightsᵀ @ ∂L/∂context_pre  →  shape: (3, 8)
         ∂L/∂weights      = ∂L/∂context_pre @ Vᵀ         →  shape: (3, 3)

3. Softmax (per row i):
         ∂L/∂scores[i] = weights[i] ⊙ ( ∂L/∂weights[i] − weights[i]·∂L/∂weights[i] )

4. Q, K: ∂L/∂Q  = ∂L/∂scores @ K / √8                   →  shape: (3, 8)
         ∂L/∂K  = ∂L/∂scoresᵀ @ Q / √8                  →  shape: (3, 8)

5. WQ, WK, WV:
         ∂L/∂WQ2 = H1_lnᵀ @ ∂L/∂Q                       →  shape: (8, 8)
         ∂L/∂WK2 = H1_lnᵀ @ ∂L/∂K                       →  shape: (8, 8)
         ∂L/∂WV2 = H1_lnᵀ @ ∂L/∂V                       →  shape: (8, 8)

6. Back to H1_ln:
         ∂L/∂H1_ln = ∂L/∂Q @ WQ2ᵀ + ∂L/∂K @ WK2ᵀ + ∂L/∂V @ WV2ᵀ
                   →  shape: (3, 8)
```

---

### Step B6 — Block 1 (same pattern, own weights)

Gradient flows through:

```
layernorm(H1)  →  mlp_backward(W1_1, W2_1)  →  layernorm(A1)  →  attention_backward(WQ1, WK1, WV1, WO1)
```

This produces `∂L/∂X` of shape **(3, 8)** — the gradient w.r.t. the input embeddings.

---

### Step B7 — Gradient through Embeddings

For each token in the input sequence, update its embedding row:

```
embedding_matrix[vocab["I"]]    -= lr · ∂L/∂X[0]
embedding_matrix[vocab[" "]]    -= lr · ∂L/∂X[1]
embedding_matrix[vocab["like"]] -= lr · ∂L/∂X[2]
```

The embedding for `"like"` gets nudged in the direction that makes it easier for the attention layers downstream to predict `" "` after it.

---

## WEIGHT UPDATE

After backprop computes gradients for all 13 weight matrices, SGD applies:

```
W ← W − lr · ∂L/∂W       (lr = 0.01)
```

Applied to every matrix:

```
W_out  (8×10)     W1_1, W2_1  (8×16, 16×8)     W1_2, W2_2  (8×16, 16×8)
WQ1, WK1, WV1, WO1  (8×8 each)
WQ2, WK2, WV2, WO2  (8×8 each)
embedding_matrix    (10×8)
```

**Total parameters updated per step: 1,184**

---

## COMPLETE DATA FLOW SUMMARY

```
Input: ["I", " ", "like"]   Target: " " (ID 4)

FORWARD
──────────────────────────────────────────────────────────────────────
token IDs  [0, 4, 1]
    │  E (10×8)
    ▼
X          (3, 8)   ← 3 token embeddings
    │  WQ1,WK1,WV1,WO1 (8×8 each)
    ▼
A1         (3, 8)   ← after attention block 1
    ▼  layernorm
A1_ln      (3, 8)
    │  W1_1 (8×16), W2_1 (16×8)
    ▼
H1         (3, 8)   ← after MLP block 1
    ▼  layernorm
H1_ln      (3, 8)
    │  WQ2,WK2,WV2,WO2 (8×8 each)
    ▼
A2         (3, 8)   ← after attention block 2
    ▼  layernorm
A2_ln      (3, 8)
    │  W1_2 (8×16), W2_2 (16×8)
    ▼
H2         (3, 8)   ← after MLP block 2
    ▼  layernorm
H2_ln      (3, 8)
    │  W_out (8×10)
    ▼
logits     (3, 10)
    ▼  logits[-1]
last_logits (10,)
    ▼  softmax
probs       (10,)   →   L = -log(probs[4])   ≈ 2.12

BACKWARD
──────────────────────────────────────────────────────────────────────
grad_logits  (10,)   = probs − one_hot(4)
    │  W_out
    ▼
∂L/∂H2_ln  (3, 8)   ← last row only
    │  layernorm
    ▼
∂L/∂H2     (3, 8)
    │  W1_2, W2_2, ReLU
    ▼
∂L/∂A2_ln  (3, 8)   →   grads: ∂W1_2, ∂W2_2
    │  layernorm
    ▼
∂L/∂A2     (3, 8)
    │  WQ2,WK2,WV2,WO2, softmax
    ▼
∂L/∂H1_ln  (3, 8)   →   grads: ∂WQ2, ∂WK2, ∂WV2, ∂WO2
    │  layernorm
    ▼
∂L/∂H1     (3, 8)
    │  W1_1, W2_1, ReLU
    ▼
∂L/∂A1_ln  (3, 8)   →   grads: ∂W1_1, ∂W2_1
    │  layernorm
    ▼
∂L/∂A1     (3, 8)
    │  WQ1,WK1,WV1,WO1, softmax
    ▼
∂L/∂X      (3, 8)   →   grads: ∂WQ1, ∂WK1, ∂WV1, ∂WO1
    │  embedding lookup
    ▼
embedding_matrix rows [0, 4, 1] updated in-place

UPDATE:   W ← W − 0.01 · ∂L/∂W   for all 13 matrices + embeddings
```

---

## WHY 400 EPOCHS?

One update on one training pair nudges 1,184 parameters by a tiny amount. The dataset has 5 sentences × ~7 pairs each = **35 training pairs per epoch**. The model must learn:

- Sentences starting with `"I"` are followed by `" "` then `"like"` or `"hate"`
- `"like"/"hate"` are always followed by an animal word
- `"birds"` only follows `"hate"` — never `"like"`

Each of these patterns lives across multiple weight matrices. It takes hundreds of passes before the gradients consistently reinforce the same directions across all 1,184 parameters simultaneously.

---

## WHAT EACH LAYER IS ACTUALLY LEARNING

| Layer | What gradient adjusts it toward |
|-------|----------------------------------|
| Embeddings | `"like"` and `"hate"` land in similar regions (both precede animals); `"cats"` and `"dogs"` cluster together |
| WQ1, WK1 | Which token pairs to match up — e.g. verbs should query their subject |
| WV1, WO1 | What information to extract and forward when a match is found |
| W1_1, W2_1 | Pattern detectors: one hidden unit might activate for "verb following subject" |
| Block 2 weights | Higher-order patterns built on Block 1's abstractions |
| W_out | Maps the final 8-dim representation to a score over the 10 output tokens |
