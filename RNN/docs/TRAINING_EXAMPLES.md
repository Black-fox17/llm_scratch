# Training Output Example

Here's what you'll see during training with the new sampling feature:

```
Using device: cpu
Vocabulary size: 73
================================================================================
Epoch 0 | Step 0 | Loss: 4.2891
Epoch 0 | Step 100 | Loss: 3.1245
Epoch 0 | Step 200 | Loss: 2.8934
Epoch 0 | Step 300 | Loss: 2.6721
...
Epoch 0 | Step 900 | Loss: 2.1456
================================================================================
📊 EPOCH 0 COMPLETE | Average Loss: 2.5432
================================================================================

🎨 SAMPLE GENERATIONS:
--------------------------------------------------------------------------------

Seed: 'The '
Generated: The xjk lqp mnr vwz asd fgh...   [Random gibberish - Model hasn't learned yet]

Seed: 'Hello'
Generated: Helloxyz abc def pqr...          [Random gibberish]

Seed: 'What '
Generated: What mno pqr stu vwx...          [Random gibberish]

Seed: 'I '
Generated: I klm nop qrs tuv...              [Random gibberish]

================================================================================

💾 Saved checkpoint for epoch 0

Epoch 1 | Step 0 | Loss: 2.0123
...

================================================================================
📊 EPOCH 1 COMPLETE | Average Loss: 1.9834
================================================================================

🎨 SAMPLE GENERATIONS:
--------------------------------------------------------------------------------

Seed: 'The '
Generated: The cat the the the...           [Starting to repeat learned words]

Seed: 'Hello'
Generated: Hellond the cat is the...        [Making progress]

--------------------------------------------------------------------------------
... LATER EPOCHS ...
--------------------------------------------------------------------------------

📊 EPOCH 10 COMPLETE | Average Loss: 0.8234
================================================================================

🎨 SAMPLE GENERATIONS:
--------------------------------------------------------------------------------

Seed: 'The '
Generated: The quick brown fox jumps over the lazy dog. This is a wonderful day.

Seed: 'Hello'
Generated: Hello! How are you doing today? I hope you're having a great time.

Seed: 'What '
Generated: What is your favorite color? Mine is blue like the ocean waters.

Seed: 'I '
Generated: I enjoy reading books and learning new things every single day.

================================================================================

💾 Saved checkpoint for epoch 10
```

## What to Watch For:

### **Epoch 0-2**: 
- 🔴 **Random gibberish**: The model hasn't learned patterns yet
- Loss: 2.5-4.0

### **Epoch 3-5**: 
- 🟡 **Repeated words**: Model learns common words like "the", "and"
- Loss: 1.5-2.5

### **Epoch 6-10**: 
- 🟢 **Short phrases**: Coherent 2-3 word combinations appear
- Loss: 1.0-1.5

### **Epoch 10-15**: 
- 🔵 **Sentences**: Grammatically correct sentences
- Loss: 0.7-1.0

### **Epoch 15-20**: 
- 💚 **Context-aware text**: Meaningful completions
- Loss: 0.5-0.7

## Benefits of This Approach:

✅ **Visual Progress**: See improvement in real-time  
✅ **Early Stopping**: Stop if quality is good enough  
✅ **Debugging**: Detect if model is stuck or overfitting  
✅ **Motivation**: Watching it learn is exciting!  
✅ **Temperature Testing**: See how randomness affects output
