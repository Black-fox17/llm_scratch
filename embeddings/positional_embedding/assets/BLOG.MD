# Visualizing Positional Encodings: What Actually Changes Inside a Transformer

Transformers are a type of neural network used for natural language processing (NLP). They came as a huge advancement to the field of NLP and became the go-to model for NLP tasks like machine translation, text summarization, question answering, and more—replacing the previous state of the art(SOTA) models like Recurrent Neural Networks and Long Short-Term Memory networks.

In previous models, they were sequential models which means they processed words one by one and memorized the order of the sequence. Transformers do this differently: they process words in parallel and compute attention for each word. This makes them permutation invariant—to the model, the sequence is just a cloud of data points.

If we don't tell the transformer model where each word is located, sentences like "The cat sat on the chair" will be the same as "The chair on sat the cat," or "The man bit the dog" and "The dog bit the man." We can notice the incoherent manner of these sentences and how much position matters.

![A visualization of randomly sampled tokens without positional encoding](without_positions.gif)

Transformers fix this order problem by introducing **Positional Encoding** to their architecture. This ensures order is added to the token before attention is being computed.

In the transformer architecture released by Vaswani et al. in 2017 in a paper titled "Attention is All You Need," they made use of **Sinusoidal Positional Encoding**, which we'll get to in a second.

Over the years, researchers in the field have found faults in each approach, particularly with the problem of **extrapolation**—how it performs on unseen data from its training and how it could scale in sequence length. This has led to many techniques for doing positional encoding.

In this post, I'll be discussing four of these with visualization:

1. **Sinusoidal**
2. **Learned Embeddings**
3. **RoPE (Rotary Positional Embedding)**
4. **ALiBi (Attention with Linear Biases)**

---

## 1. Sinusoidal Positional Encoding

Sinusoidal, like I said, was introduced in 2017 in the original transformer paper as the go-to way for positional encoding, and there was a major reason behind choosing it. It was intended to support extrapolation, but later work showed that absolute encodings degrade in attention quality at longer contexts compared to relative methods.

The idea behind it was pretty simple. It follows a basic deterministic computation, i.e., for a given `seq_length` and `d_model`, compute the position embeddings using the formula:

**For even positions:**

$$PE_{pos, 2i} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**For odd positions:**

$$PE_{pos, 2i+1} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- **pos**: The position of the word in the sentence (for example, 0 for the first word, 1 for the second, and so on)
- **i**: The dimension index of the embedding vector, maps to column index. `2i` will indicate an even position and `2i+1` will indicate an odd position
- **d_model**: The predefined dimensionality of the token embeddings (for example, 512)
- **10000**: A user-defined scalar value
- **PE**: Position function for mapping position `pos` in the input sequence to get the positional mapping

The reason behind this approach is because sin and cosine are periodic functions. They are smooth and continuous: small changes in input result in small changes in output, which gives us a way to represent positions in a differentiable space and prevents explosion of signals, constraining the values between [-1, 1].

By varying the frequency of the wavelengths across dimensions, we can create a rich, multiscale representation of position. Adding these computed deterministic values to our tokens removes the noise in our visualization as seen above.

You can see this in the visualization below:

![Sinusoidal Visualization 3D](sinusoidal_visualization_3d.gif)

The values computed from the formula are added to our word token embeddings, contributing to its absolute nature. This leads to an additional task for the model to unmix the position from the semantic meaning.

Research has shown relative encoding works best at positional encoding compared to absolute.

---

## 2. Learned Embeddings

Learned embeddings was an alternative to the sinusoidal PE. It was an approach released alongside sinusoidal in the hope to **LEARN** position. This makes it a stochastic model, different from how our sinusoidal model works.

The idea sounds pretty decent. As we know, a trained model will definitely perform better than a computed function in learning complex relationships. This was proven true—not only were positions in sequences learned during training, but it also gives the model more flexibility to memorize positional patterns during training. For example, a word that appears at the beginning of a sequence has a relationship to the word that appears at the end. 

Well, this does sound really great and will contribute to the performance of the model. You might ask: why is this not the go to approach to PE since it performs better than sinusoidal in capturing relationships and was invented by the same Vaswani et al. team?

The simple answer here is **it couldn't extrapolate to a longer input sequence at all**.

This means it's performance drops sharply once inference exceeds the maximum trained sequence length.

This was a major setback for it. In the world of ML, we hope to prevent overfitting, and our model is only as good as how it performs on unseen data during training.

---

## 3. RoPE (Rotary Positional Embedding)

RoPE, also known as **Rotary Positional Embedding**, solves position by geometry. This was a major breakthrough in relative positioning and a major step up from absolute position like sinusoidal, where the model has to learn to separate semantics from position.

The concept of its relative positioning allows the model to extrapolate for longer sequences.

Its concept is very simple yet intuitive: it applies a 2D rotation matrix to each pair of embedding space. This is achieved by defining a complex-valued representation for each token embedding and then rotating it based on token position.

### The Concept Behind Relative Positioning

Imagine a needle on a compass:
- In **Sinusoidal PE**, you would move the needle by pushing it to a new location.
- In **RoPE**, you rotate the needle by a specific angle.

The magic is that if word A is at position 2 and word B is at position 5, the "angle" between them is always the same (3 units of rotation), no matter where they are in the sentence.

### A Walkthrough Example

Let's simplify. Imagine your word embedding has only 2 dimensions $(x, y)$.

- **The Word**: You have the word "Apple." Its embedding is $[1, 0]$.
- **The Position**: It is at Position 1.
- **The Rotation**: We decide that each position rotates the vector by $90°$.

At Pos 1: $[1, 0]$ rotated $90° \rightarrow [0, 1]$  
At Pos 2: $[1, 0]$ rotated $180° \rightarrow [-1, 0]$  
At Pos 3: $[1, 0]$ rotated $270° \rightarrow [0, -1]$

When the attention mechanism calculates the "closeness" (dot product) between two words, it only cares about the difference in their angles. This is why RoPE is so good at understanding relative distance.

### The Math

**1. Define token representation:**

Each token embedding $x \in \mathbb{R}^d$ is split into two interleaved parts:

$$x = (x_0, x_1, x_2, x_3, ..., x_{d-2}, x_{d-1})$$

Each consecutive pair $x_{2i}$ and $x_{2i+1}$ can be interpreted as a complex number:

$$\tilde{x}_i = x_{2i} + ix_{2i+1} \quad \text{where } i = \sqrt{-1}$$

**2. Each pair is rotated by an angle:**

$$\theta_{pos,i} = pos \cdot \omega_i$$

where

$$\omega_i = 10000^{-2i/d}$$

**3. Rotation:**

$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

Expanding it, we have:

$$x'_{2i} = x_{2i} \cos\theta - x_{2i+1} \sin\theta$$
$$x'_{2i+1} = x_{2i} \sin\theta + x_{2i+1} \cos\theta$$

In complex form:

$$\tilde{x}'_i = x_{2i} \cos\theta - x_{2i+1} \sin\theta + i(x_{2i} \sin\theta + x_{2i+1} \cos\theta)$$

**4. Combine back:**

$$x' = (x'_{2i}, x'_{2i+1}, x'_{2i+2}, x'_{2i+3}, ..., x'_{d-2}, x'_{d-1})$$

That's just the math. We use the rotary matrix to rotate each pair of elements.

In the self-attention mechanism, we rotate the Q and K to be Q' and K', and attention mechanism is computed as usual:

$$A = \frac{Q'K'^T}{\sqrt{d}}$$

After rotation, the dot product between Q and K depends only on their relative position difference.

You can see the visualization below of how it performs position with the rotation of embeddings:

![RoPE Visualization](with_rope_visualization.gif)

---

## 4. ALiBi (Attention with Linear Biases)

**Standard attention:**

$$\text{score}(i,j) = \frac{q_i^\top k_j}{\sqrt{d}}$$

**ALiBi attention:**

$$\text{score}(i,j) = \frac{q_i^\top k_j}{\sqrt{d}} - m_h \cdot |i - j|$$

That's it.

Where:
- **i** = query position
- **j** = key position
- **m_h** = head-specific slope
In practice, slopes are deterministically assigned so earlier heads attend locally while later heads attend globally
**Far tokens → lower score**  
**Near tokens → higher score**

No embeddings. No rotations.

Self-attention already computes similarity. ALiBi just says:

> "Closer tokens should be preferred, linearly."

### The Bias

The bias:
- Is **relative** (depends only on distance)
- Each attention head gets a different slope:
  - **Small slope** → long-range attention
  - **Large slope** → local attention

**Typical construction (conceptual):**

```
head 0: m = very small
head 1: m = small
...
head N: m = large
```

So the model naturally learns:
- **Syntax** (local heads)
- **Semantics / topic** (global heads)

Without learning positional parameters.

It extrapolates well for longer sequences too. No tables to overflow. No frequencies to alias—just a simple nudge in the attention computation.

The idea works well. It allows us to introduce a simple bias to the attention score to nudge the model's performance towards position and could extrapolate well.

---

## Conclusion

Currently in the industry, **RoPE and ALiBi dominate modern large language models but Rope is more preffered in the industry** because of its approach with geometry. ALiBi, though it works well on large contextual lengths, introduces a bias to the attention strength, whereas in RoPE we are rotating the vector in the dimension space and showing rotational and complex relationships better than any technique.

And we can also achieve the same extrapolation found in ALiBi with **NTK-aware scaled RoPE**.

Other techniques also exist, like **NoPE** (No Positional Encoding) you might want to check it out!