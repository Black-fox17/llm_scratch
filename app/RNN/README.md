# LSTM Character-Level Language Model & Chatbot

A from-scratch implementation of a stacked LSTM network for character-level language modeling, with an interactive chatbot interface.

## 📁 Project Structure

```
RNN/
├── models/
│   ├── cell.py           # LSTM cell implementation
│   └── stacked.py        # Stacked LSTM model
├── data/
│   ├── batchloader.py    # Character-level data loader
│   └── input.txt         # Training text data
├── train.py              # Training script
├── generate.py           # Text generation utilities
├── chatbot.py            # Interactive chatbot
├── checkpoints/          # Saved model checkpoints (created during training)
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch
```

### 2. Prepare Training Data

Add your text data to `data/input.txt`. The provided sample text is a good starting point, but you can replace it with any text corpus you'd like the model to learn from.

### 3. Train the Model

```bash
python train.py
```

**Training Configuration:**
- Vocabulary: Character-level (all unique characters in input.txt)
- Hidden size: 128
- Number of layers: 2
- Sequence length: 50
- Batch size: 50
- Learning rate: 2e-3
- Gradient clipping: 5.0
- Epochs: 20

The training will:
- Display loss every 100 steps
- **Generate sample text after each epoch** to monitor learning progress
- Save checkpoints after each epoch in the `checkpoints/` directory
- Show progress with format: `Epoch X | Step Y | Loss: Z.ZZZZ`

**Sample Output After Each Epoch:**
```
================================================================================
📊 EPOCH 0 COMPLETE | Average Loss: 2.5432
================================================================================

🎨 SAMPLE GENERATIONS:
--------------------------------------------------------------------------------

Seed: 'The '
Generated: The quick brown...

Seed: 'Hello'
Generated: Hello! How are you...
```

See `TRAINING_EXAMPLES.md` for detailed examples of how the model improves over epochs.

### 4. Generate Text

Test the model's text generation:

```bash
python generate.py
```

This will generate text samples starting from various seed texts.

### 5. Run the Chatbot

Start the interactive chatbot:

```bash
python chatbot.py
```

**Chatbot Commands:**
- Type normally to chat with the bot
- `help` - Show available commands
- `temp <0.1-2>` - Adjust temperature (e.g., `temp 0.9`)
- `quit`, `exit`, or `q` - Exit the chatbot

**Temperature Guide:**
- **0.1-0.5**: Conservative, repetitive outputs
- **0.6-0.9**: Balanced, recommended range
- **1.0-1.5**: Creative, more random
- **1.6-2.0**: Very random, experimental

## 🧠 How It Works

### LSTM Cell (`models/cell.py`)

Implements the core LSTM operations:
- **Input Gate (i)**: Controls what new information to add
- **Forget Gate (f)**: Controls what information to discard
- **Cell Gate (g)**: Generates candidate values
- **Output Gate (o)**: Controls what to output

```
c_next = f * c + i * g
h_next = o * tanh(c_next)
```

### Stacked LSTM (`models/stacked.py`)

- Embedding layer: Converts character indices to dense vectors
- Multiple LSTM layers: Process sequences with hidden states
- Output projection: Maps hidden states to vocabulary probabilities

### Training (`train.py`)

Uses truncated backpropagation through time (TBPTT):
1. Initialize hidden and cell states
2. Process sequences character by character
3. Compute cross-entropy loss
4. Backpropagate and update weights
5. Detach states to prevent gradients flowing too far back

### Generation (`generate.py`)

- **Sampling**: Sample characters from the probability distribution
- **Temperature**: Controls randomness (higher = more random)
- **State Management**: Maintains LSTM hidden states across generation steps

### Chatbot (`chatbot.py`)

- Loads trained model from latest checkpoint
- Processes user input as prompt
- Generates response using the trained LSTM
- Interactive loop with command support

## 📊 Model Details

**Parameters:**
- Embedding: `vocab_size × hidden_size`
- LSTM weights per layer: `(input_size + hidden_size) × (4 × hidden_size)`
- Output projection: `hidden_size × vocab_size`

**Total trainable parameters:** ~100K-500K (depending on vocabulary size)

## 💡 Tips for Better Results

1. **More training data**: Add more text to `data/input.txt`
2. **Longer training**: Increase epochs in `train.py`
3. **Larger model**: Increase `hidden_size` or `num_layers`
4. **Adjust temperature**: Experiment with different values during generation
5. **Better data**: Use domain-specific text for focused responses

## 🔧 Customization

### Change Model Architecture

Edit `train.py`:
```python
hidden_size = 256  # Increase for more capacity
num_layers = 3     # Add more layers
```

### Adjust Training

Edit `train.py`:
```python
lr = 1e-3          # Lower learning rate for stability
batch_size = 64    # Larger batches (requires more memory)
seq_len = 100      # Longer sequences
```

### Modify Generation

Edit `generate.py` or `chatbot.py`:
```python
temperature = 0.8   # Default temperature
max_length = 300    # Longer generations
```

## 🐛 Troubleshooting

**"No checkpoints found"**
- Run `python train.py` first to create model checkpoints

**"CUDA out of memory"**
- Reduce `batch_size` or `hidden_size` in `train.py`
- The model will automatically use CPU if CUDA is unavailable

**Poor quality responses**
- Train for more epochs
- Add more diverse training data
- Adjust temperature during generation
- Increase model size

**Import errors**
- Ensure all files are in correct directories
- Run from the `RNN/` directory: `python train.py`

## 📝 Example Usage

### Training Output
```
Using device: cpu
Vocabulary size: 67
Epoch 0 | Step 0 | Loss: 4.2031
Epoch 0 | Step 100 | Loss: 2.8754
Epoch 0 | Step 200 | Loss: 2.1234
...
Saved checkpoint for epoch 0
```

### Chatbot Interaction
```
You: What is your favorite color?
Bot: Mine is blue like the ocean and sky.

You: Tell me about neural networks
Bot: They can learn patterns from data and process information.
```

## 🎯 Next Steps

- [ ] Add validation loss tracking
- [ ] Implement beam search for generation
- [ ] Add attention mechanism
- [ ] Support word-level tokenization
- [ ] Add model evaluation metrics
- [ ] GPU optimization

## 📚 Learning Resources

This implementation demonstrates:
- ✅ Manual backpropagation (no `nn.Module`)
- ✅ LSTM cell mechanics
- ✅ Truncated backpropagation through time
- ✅ Gradient clipping
- ✅ Character-level language modeling
- ✅ Text generation with sampling
- ✅ Interactive chatbot design

---

**Author**: Built from scratch for educational purposes  
**License**: MIT
