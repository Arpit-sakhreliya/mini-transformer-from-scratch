import numpy as np

# Vocabulary
vocab = {
    "I": 0,
    "like": 1,
    "cats": 2,
    "dogs": 3,
    " ": 4,
    "you": 5,
    "hate": 6,
    "birds": 7,
    ".": 8,
    "<END>": 9
}

# Reverse lookup
id_to_token = {v: k for k, v in vocab.items()}

# Embedding settings
vocab_size = len(vocab)      # 10
embedding_dim = 8

# Random initialization
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)


# =========================
# STEP 2: Tiny Attention Layer
# =========================

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def layernorm(x, eps=1e-5):
    return (x - x.mean()) / (x.std() + eps)

d = embedding_dim  # 8


np.random.seed(1)
WQ1 = np.random.randn(d, d)
WK1 = np.random.randn(d, d)
WV1 = np.random.randn(d, d)
WO1 = np.random.randn(d, d)

np.random.seed(2)
WQ2 = np.random.randn(d, d)
WK2 = np.random.randn(d, d)
WV2 = np.random.randn(d, d)
WO2 = np.random.randn(d, d)


def attention_forward(X, WQ, WK, WV, WO):
    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    seq_len = X.shape[0]
    d = X.shape[1]

    scores = (Q @ K.T) / np.sqrt(d)
    weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    output_pre_WO = weights @ V
    output = output_pre_WO @ WO

    cache = {
        "X": X,
        "Q": Q,
        "K": K,
        "V": V,
        "weights": weights,
        "output_pre_WO": output_pre_WO
    }

    return output, cache

def embed(sentence):
    return np.array([embedding_matrix[vocab[word]] for word in sentence])


# =========================
# STEP 3: Tiny Neural Network (MLP)
# =========================

def relu(x):
    return np.maximum(0, x)

def mlp_forward(X, W1, W2):
    hidden_pre_relu = X @ W1
    hidden = relu(hidden_pre_relu)
    output = hidden @ W2

    cache = {
        "X": X,
        "hidden_pre_relu": hidden_pre_relu,
        "hidden": hidden,
        "W1": W1,
        "W2": W2
    }

    return output, cache


# =========================
# STEP 4: OUTPUT LAYER
# =========================

np.random.seed(4)
W_out = np.random.randn(d, vocab_size)

def output_layer(X):
    return X @ W_out

def predict_next_token(logits, temperature=1.0):
    last_logits = logits[-1]
    scaled_logits = last_logits / temperature
    exps = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exps / np.sum(exps)
    return np.argmax(probs), probs


# =========================
# STEP 5: Dataset and training pairs
# =========================

dataset = [
    ["I", " ", "like", " ", "cats", " ", ".", "<END>"],
    ["I", " ", "like", " ", "dogs", " ", ".", "<END>"],
    ["you", " ", "hate", " ", "birds", " ", ".", "<END>"],
    ["you", " ", "like", " ", "cats", " ", ".", "<END>"],
    ["I", " ", "hate", " ", "birds", " ", ".", "<END>"]
]

def create_training_pairs(sentence):
    pairs = []
    for i in range(1, len(sentence)):
        input_seq = sentence[:i]
        target = sentence[i]
        pairs.append((input_seq, target))
    return pairs


# =========================
# STEP 6: Full Forward Pass + Loss
# =========================

# --- FIX: single definition of model_forward, using block-specific weights ---
def model_forward(input_seq):
    X = embed(input_seq)

    # Block 1
    A1, attn1_cache = attention_forward(X, WQ1, WK1, WV1, WO1)
    A1_ln = layernorm(A1)
    H1, mlp1_cache = mlp_forward(A1_ln, W1_1, W2_1)
    H1_ln = layernorm(H1)

    # Block 2
    A2, attn2_cache = attention_forward(H1_ln, WQ2, WK2, WV2, WO2)
    A2_ln = layernorm(A2)
    H2, mlp2_cache = mlp_forward(A2_ln, W1_2, W2_2)
    H2_ln = layernorm(H2)

    logits = output_layer(H2_ln)

    cache = {
        "X": X,

        "attn1": attn1_cache,
        "mlp1": mlp1_cache,

        "attn2": attn2_cache,
        "mlp2": mlp2_cache,

        "A1": A1, "A1_ln": A1_ln,
        "H1": H1, "H1_ln": H1_ln,

        "A2": A2, "A2_ln": A2_ln,
        "H2": H2, "H2_ln": H2_ln
    }

    return logits, cache

def softmax_probs(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

def cross_entropy_loss(logits, target_id):
    probs = softmax_probs(logits)
    loss = -np.log(probs[target_id] + 1e-9)
    return loss, probs


# =========================
# STEP 7: Backpropagation
# =========================

learning_rate = 0.01

# --- FIX: return 4 values (loss, dW_out, dH2, probs) so caller can unpack correctly ---
def output_layer_backward(H2_ln, logits, target_id):
    last_logits = logits[-1]
    exp_logits = np.exp(last_logits - np.max(last_logits))
    probs = exp_logits / np.sum(exp_logits)

    loss = -np.log(probs[target_id] + 1e-9)

    # dL/dlogits = probs - one_hot(target)
    grad_logits = probs.copy()
    grad_logits[target_id] -= 1

    # logits = H2_ln @ W_out  →  dW_out = H2_ln[-1]^T outer grad_logits
    last_hidden = H2_ln[-1]
    grad_W_out = np.outer(last_hidden, grad_logits)

    # gradient back to H2_ln (only last token position)
    grad_H2_ln = np.zeros_like(H2_ln)
    grad_H2_ln[-1] = grad_logits @ W_out.T

    return loss, grad_W_out, grad_H2_ln, probs

def layernorm_backward(grad_output, x, eps=1e-5):
    mean = x.mean()
    std = np.sqrt(x.var() + eps)
    x_hat = (x - mean) / std
    N = x.size

    grad_x = (1.0 / std) * (
        grad_output
        - np.mean(grad_output)
        - x_hat * np.mean(grad_output * x_hat)
    )
    return grad_x

# --- FIX: removed `global W1, W2` (those globals don't exist); weights live in cache ---
def mlp_backward(grad_output, cache):
    X = cache["X"]
    hidden_pre_relu = cache["hidden_pre_relu"]
    hidden = cache["hidden"]
    W1 = cache["W1"]
    W2 = cache["W2"]

    # out = hidden @ W2
    grad_W2 = hidden.T @ grad_output
    grad_hidden = grad_output @ W2.T

    # ReLU backward
    grad_hidden_pre = grad_hidden.copy()
    grad_hidden_pre[hidden_pre_relu <= 0] = 0

    # hidden_pre = X @ W1
    grad_W1 = X.T @ grad_hidden_pre
    grad_X = grad_hidden_pre @ W1.T

    return grad_X, grad_W1, grad_W2

# --- FIX: attention_backward takes explicit WQ/WK/WV/WO for that block ---
def attention_backward(grad_out, X, cache, WQ, WK, WV, WO):
    Q = cache["Q"]
    K = cache["K"]
    V = cache["V"]
    weights = cache["weights"]
    output_pre_WO = cache["output_pre_WO"]

    seq_len, d = X.shape

    # WO gradient: out = output_pre_WO @ WO
    grad_WO = output_pre_WO.T @ grad_out
    grad_attn = grad_out @ WO.T

    # V gradient: output_pre_WO = weights @ V
    grad_V = weights.T @ grad_attn
    grad_weights = grad_attn @ V.T

    # Softmax backward
    grad_scores = np.zeros_like(weights)
    for i in range(seq_len):
        w = weights[i]
        g = grad_weights[i]
        grad_scores[i] = w * (g - np.dot(w, g))

    # Q, K gradients: scores = QK^T / sqrt(d)
    grad_Q = grad_scores @ K / np.sqrt(d)
    grad_K = grad_scores.T @ Q / np.sqrt(d)

    # Projection gradients
    grad_WQ = X.T @ grad_Q
    grad_WK = X.T @ grad_K
    grad_WV = X.T @ grad_V

    grad_X = grad_Q @ WQ.T + grad_K @ WK.T + grad_V @ WV.T

    return grad_X, grad_WQ, grad_WK, grad_WV, grad_WO

def embed_backward(grad_X, sentence):
    global embedding_matrix
    for i, word in enumerate(sentence):
        idx = vocab[word]
        embedding_matrix[idx] -= learning_rate * grad_X[i]


# =========================
# MLP BLOCK 1 & 2 weights
# =========================
np.random.seed(3)
W1_1 = np.random.randn(d, 16)
W2_1 = np.random.randn(16, d)

np.random.seed(5)
W1_2 = np.random.randn(d, 16)
W2_2 = np.random.randn(16, d)


# =========================
# Full Backward Pass
# =========================

def model_backward(logits, cache, target_id, input_seq):
    H2_ln = cache["H2_ln"]

    # OUTPUT LAYER
    # --- FIX: unpack 4 values ---
    loss, dW_out, dH2_ln, probs = output_layer_backward(H2_ln, logits, target_id)

    # =========================
    # BLOCK 2 — backward
    # =========================
    # H2_ln = layernorm(H2)  →  grad back to H2
    dH2 = layernorm_backward(dH2_ln, cache["H2"])

    # H2 = mlp2(A2_ln) forward, so backward gives grad w.r.t. A2_ln
    dA2_ln, dW1_2, dW2_2 = mlp_backward(dH2, cache["mlp2"])

    # A2_ln = layernorm(A2)  →  grad back to A2
    dA2 = layernorm_backward(dA2_ln, cache["A2"])

    # A2 = attention2(H1_ln)  →  grad w.r.t. H1_ln and attn2 weights
    dH1_ln, dWQ2, dWK2, dWV2, dWO2 = attention_backward(
        dA2, cache["H1_ln"], cache["attn2"], WQ2, WK2, WV2, WO2
    )

    # =========================
    # BLOCK 1 — backward
    # =========================
    # H1_ln = layernorm(H1)  →  grad back to H1
    dH1 = layernorm_backward(dH1_ln, cache["H1"])

    # H1 = mlp1(A1_ln) forward
    dA1_ln, dW1_1, dW2_1 = mlp_backward(dH1, cache["mlp1"])

    # A1_ln = layernorm(A1)  →  grad back to A1
    dA1 = layernorm_backward(dA1_ln, cache["A1"])

    # A1 = attention1(X)
    dX, dWQ1, dWK1, dWV1, dWO1 = attention_backward(
        dA1, cache["X"], cache["attn1"], WQ1, WK1, WV1, WO1
    )

    # EMBEDDINGS
    embed_backward(dX, input_seq)

    grads = {
        "W_out": dW_out,

        "W1_1": dW1_1, "W2_1": dW2_1,
        "W1_2": dW1_2, "W2_2": dW2_2,

        "WQ1": dWQ1, "WK1": dWK1, "WV1": dWV1, "WO1": dWO1,
        "WQ2": dWQ2, "WK2": dWK2, "WV2": dWV2, "WO2": dWO2,

        "loss": loss
    }

    return grads


# =========================
# Parameter Update
# =========================

def update_params(grads, lr=0.01):
    global W_out
    global W1_1, W2_1, W1_2, W2_2
    global WQ1, WK1, WV1, WO1
    global WQ2, WK2, WV2, WO2

    W_out -= lr * grads["W_out"]

    W1_1 -= lr * grads["W1_1"]
    W2_1 -= lr * grads["W2_1"]
    W1_2 -= lr * grads["W1_2"]
    W2_2 -= lr * grads["W2_2"]

    # --- FIX: update block 1 and block 2 attention weights separately ---
    WQ1 -= lr * grads["WQ1"]
    WK1 -= lr * grads["WK1"]
    WV1 -= lr * grads["WV1"]
    WO1 -= lr * grads["WO1"]

    WQ2 -= lr * grads["WQ2"]
    WK2 -= lr * grads["WK2"]
    WV2 -= lr * grads["WV2"]
    WO2 -= lr * grads["WO2"]


# =========================
# Training Loop
# =========================

def train(epochs=20):
    for epoch in range(epochs):
        total_loss = 0

        for sentence in dataset:
            pairs = create_training_pairs(sentence)

            for inp, tgt in pairs:
                logits, cache = model_forward(inp)

                last_logits = logits[-1]
                loss, _ = cross_entropy_loss(last_logits, vocab[tgt])
                total_loss += loss

                grads = model_backward(logits, cache, vocab[tgt], inp)
                update_params(grads, lr=learning_rate)

        print(f"Epoch {epoch+1:>2} | Loss: {total_loss:.4f}")


# =========================
# EVALUATE
# =========================

def generate(input_seq, max_tokens=20):
    """
    Autoregressively predict tokens one at a time from input_seq
    until <END> is produced or max_tokens is reached.
    Returns the full sequence (input + generated tokens).
    """
    current_seq = list(input_seq)

    for _ in range(max_tokens):
        logits, _ = model_forward(current_seq)
        pred_id, _ = predict_next_token(logits)
        next_token = id_to_token[pred_id]
        current_seq.append(next_token)

        if next_token == "<END>":
            break

    return current_seq


def evaluate(n=5):
    """
    Randomly pick n sentences from the dataset.
    For each, use only the first token as the prompt and
    let the model generate the rest autoregressively.
    Prints prompt, full generated output, expected output, and match result.
    """
    indices = np.random.choice(len(dataset), size=n, replace=False)
    samples = [dataset[i] for i in indices]

    print("\n" + "=" * 52)
    print("                  EVALUATION")
    print("=" * 52)

    for i, sentence in enumerate(samples):
        prompt = sentence[:1]
        generated = generate(prompt)

        generated_tokens = generated[len(prompt):]
        expected_tokens  = sentence[1:]

        prompt_str    = "".join(prompt)
        generated_str = "".join(generated_tokens)
        expected_str  = "".join(expected_tokens)
        correct       = generated_tokens == expected_tokens

        print(f"\n  Sample {i+1}")
        print(f"  Input token : {prompt_str!r}")
        print(f"  Generated   : {prompt_str + generated_str!r}")
        print(f"  Expected    : {prompt_str + expected_str!r}")
        print(f"  Match       : {'✓  correct' if correct else '✗  wrong'}")

    print("\n" + "=" * 52 + "\n")

evaluate() 
train(epochs=400)
evaluate()