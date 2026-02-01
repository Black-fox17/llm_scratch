
import torch
import torch.nn.functional as F
import torch.optim as optim

class SkipGramTrainer:
    """
    Handles the training loops for Skip-Gram Word2Vec.
    Includes two methods:
    1. Full Softmax (Reference implementation using standard PyTorch)
    2. Negative Sampling (Optimized implementation using manual gradient updates)
    """
    def __init__(self, dataset, embedding_dim=10, device=None):
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.device = device if device else torch.device("cpu")
        
        # Model Parameters (Weights)
        # V x D and D x V
        self.W_in = None
        self.W_out = None
    
    def initialize_weights(self):
        V = self.dataset.tokenizer.vocab_size
        if V == 0:
            raise ValueError("Vocabulary is empty. Cannot initialize weights.")
            
        # W_in: Center word embeddings
        # W_out: Generally considered 'Context' embeddings
        self.W_in = torch.randn(V, self.embedding_dim, device=self.device, requires_grad=True)
        self.W_out = torch.randn(self.embedding_dim, V, device=self.device, requires_grad=True)

    def train_full_softmax(self, num_epochs=1000, lr=0.01, lr_decay=0.99):
        """
        Part 1: Full Softmax Training.
        
        Math:
            P(context | center) = softmax(v_center . W_out)
            Loss = -log(P(true_context | center))
            
        This computes the denominator over the ENTIRE vocabulary sum(exp(...)),
        which is slow for large vocabs but great for understanding.
        """
        print("\n--- Starting Full Softmax Training ---")
        self.initialize_weights()
        
        optimizer = optim.SGD([self.W_in, self.W_out], lr=lr)
        
        for epoch in range(num_epochs):
            loss_val = 0
            
            for center, context in self.dataset.all_pairs:
                # 1. Prepare Target
                y_true = torch.tensor([context], dtype=torch.long, device=self.device)
                
                # 2. Forward Pass
                z1 = self.W_in[center]       # (Dim,)
                z2 = z1 @ self.W_out         # (Vocab,) -> Dots with all context words
                
                # 3. Compute Loss
                log_softmax = F.log_softmax(z2, dim=0)
                loss = F.nll_loss(log_softmax.view(1, -1), y_true)
                loss_val += loss.item()

                # 4. Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Simple LR Decay
            if epoch % 50 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
            
            if epoch % 100 == 0:
                avg_loss = loss_val / max(1, len(self.dataset.all_pairs))
                print(f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}")
                
        print("Full Softmax Training Complete.")
        return self.W_in # Return for visualization

    def train_negative_sampling(self, num_epochs=2000, lr=0.01, lr_decay=0.99, neg_samples=5):
        """
        Part 2: Negative Sampling Training.
        
        Math:
            Maximize log P(Pos) + sum(log(1 - P(Neg)))
            
        Updates are sparse! We only touch the columns in W_out corresponding to
        the positive context word and the K negative samples.
        """
        print("\n--- Starting Negative Sampling Training (Manual Gradients) ---")
        self.initialize_weights()
        
        for epoch in range(num_epochs):
            total_loss = 0.0

            for center, context in self.dataset.all_pairs:
                v_c = self.W_in[center] 
                
                # ---------- Positive sample ----------
                # Sigmoid(v_c . v_context)
                score_pos = torch.sigmoid(v_c @ self.W_out[:, context])
                loss_pos = -torch.log(score_pos + 1e-8)
                total_loss += loss_pos.item()

                # Gradient rule: grad = lr * (label - probability)
                # Label is 1 for positive.
                # update = alpha * (1 - score)
                grad_pos = lr * (1 - score_pos)
                
                v_c_data = v_c.data.clone() # Snapshot for update
                
                # Update (Gradient Ascent on Log Likelihood -> or Descent on Negative Log Likelihood)
                # Here we add gradient * input because we want to Move Towards target.
                self.W_out[:, context].data += grad_pos * v_c_data
                self.W_in[center].data      += grad_pos * self.W_out[:, context].data

                # ---------- Negative samples ----------
                negs = torch.multinomial(self.dataset.unigram_dist, neg_samples, replacement=True)
                for neg in negs:
                    # Sigmoid(- v_c . v_neg)  => We want this close to 1
                    # Equivalent to Sigmoid(score) close to 0 if score = v_c . v_neg
                    
                    score_dot = v_c @ self.W_out[:, neg]
                    score_neg_prob = torch.sigmoid(score_dot) # Probability it IS context (we want 0)
                    
                    # Loss = -log(1 - prob) approx via sigmoid(-x)
                    # Let's use standard form: Score for negative is sigmoid(-dot)
                    score_neg_correct = torch.sigmoid(-score_dot)
                    loss_neg = -torch.log(score_neg_correct + 1e-8)
                    total_loss += loss_neg.item()

                    # Gradient: (0 - prob) = -prob
                    # So we subtract (lr * prob)
                    grad_neg = lr * (1 - score_neg_correct) # This is actually the magnitude based on error
                    
                    # Update: push AWAY
                    self.W_out[:, neg].data -= grad_neg * v_c_data
                    self.W_in[center].data  -= grad_neg * self.W_out[:, neg].data

            if epoch % 100 == 0 or epoch == 1:
                lr *= lr_decay
                avg_loss = total_loss / max(1, len(self.dataset.all_pairs))
                print(f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}")
        
        print("Negative Sampling Training Complete.")
        return self.W_in
