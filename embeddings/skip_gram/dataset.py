
import torch
from .simple_tokenizer import SimpleTokenizer

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

class SkipGramDataset:
    def __init__(self, source="toy", window_size=2, device=None):
        self.source = source
        self.window_size = window_size
        self.device = device if device else torch.device("cpu")
        
        self.tokenizer = SimpleTokenizer()
        self.train_corpus = []
        self.encoded_corpus = []
        self.all_pairs = []
        self.unigram_dist = None
        
        # Initialize data immediately
        self._load_data()
        self._process_data()

    def _load_data(self):
        print(f"Loading data from source: {self.source}")
        
        if self.source == "wikitext":
            if not DATASETS_AVAILABLE:
                print("Warning: 'datasets' library not found. Falling back to toy dataset.")
                self.source = "toy"
                self._load_data() # Retry with toy
                return

            try:
                # Load a small subset of wikitext for demonstration
                # Taking 1% or a small slice to keep it 'educational' and fast
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]") 
                print("Loaded Wikitext (small subset)")
                
                for item in dataset:
                    text = item['text'].strip()
                    if text:
                        self.train_corpus.append(text)
            except Exception as e:
                print(f"Error loading wikitext: {e}. Falling back to toy dataset.")
                self.train_corpus = [] # reset
                self.source = "toy" 
                self._load_data()

        else: # Default toy corpus
            self.train_corpus = [
                'drink milk', 'drink cold water', 'drink cold cola', 'drink juice',
                'drink cola', 'eat bacon', 'eat mango', 'eat cherry', 'eat apple',
                'juice with sugar', 'cola with sugar', 'mango is fruit',
                'apple is fruit', 'cherry is fruit', 'Berlin is Germany',
                'Boston is USA', 'Mercedes from Germany', 'Mercedes is a car',
                'Ford from USA', 'Ford is a car'
            ]

    def _process_data(self):
        # 1. Train Tokenizer
        print("Building vocabulary...")
        self.tokenizer.train(self.train_corpus)
        print(f"Vocabulary Size: {self.tokenizer.vocab_size}")

        # 2. Encode Corpus
        print("Encoding corpus...")
        self.encoded_corpus = [
            self.tokenizer.encode(sent) for sent in self.train_corpus
        ]
        
        # 3. Generate Pairs
        print("Generating training pairs...")
        self.all_pairs = []
        for sent_ids in self.encoded_corpus:
            if len(sent_ids) >= 2:
                self.all_pairs.extend(self._generate_pairs(sent_ids))
        print(f"Total pairs generated: {len(self.all_pairs)}")

        # 4. Prepare Unigram Distribution for Negative Sampling
        # (freq ^ 0.75) / sum
        if self.tokenizer.vocab_size > 0:
            counts = torch.tensor(
                [self.tokenizer.word_counts[self.tokenizer.id2word[i]] for i in range(self.tokenizer.vocab_size)],
                dtype=torch.float,
                device=self.device
            )
            self.unigram_dist = counts ** 0.75
            self.unigram_dist /= self.unigram_dist.sum()

    def _generate_pairs(self, sentence_ids):
        pairs = []
        n = len(sentence_ids)
        for i, center in enumerate(sentence_ids):
            start = max(0, i - self.window_size)
            end = min(n, i + self.window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((center, sentence_ids[j]))
        return pairs
