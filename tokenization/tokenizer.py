import regex as re
class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        self.encoder = vocab
        self.decoder = {v:k for k,v in vocab.items()}
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}
        self.bpe_ranks = {pair:i for i, pair in enumerate(merges)}
        self.cache = {}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bytes_to_unicode(self):
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a signficant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def get_pairs(self, word):
        """
        Return set of symbol pairs in a word.
        e.g Hello -> {(He, el), (el, lo), (He, llo), (Hel, lo)}

        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def merge_pair(self, pair, word):
        """
        merge a pair of symbols in a word.
        e.g (H,e) ,Hello -> [He, l, l, o]
        """
        merged = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                merged.append(word[i] + word[i+1])
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return merged

    def train(self, corpus, vocab_size=5000):
        """
        Train the tokenizer on a corpus of text.
        """
        # Step 1: Pre-tokenize & convert to bytes
        word_freqs = defaultdict(int)
        for text in corpus:
            for word in re.findall(self.pat, text):
                byte_word = tuple(self.byte_encoder[b] for b in word.encode("utf-8"))
                word_freqs[byte_word] += 1

        # Step 2: Initialize vocab with all single symbols
        vocab = {sym: i for i, sym in enumerate({s for w in word_freqs for s in w})}
        merges = {}

        # Step 3: Iteratively merge most frequent pairs
        while len(vocab) < vocab_size:
            # Count pair frequencies
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                pairs = get_pairs(word)
                for pair in pairs:
                    pair_freqs[pair] += freq
            if not pair_freqs:
                break

            # Most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Merge pair in all words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = merge_pair(best_pair, list(word))
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

            # Add new token to vocab
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)
            merges[best_pair] = new_token

        return vocab, merges

    def tokenize(self, token):
        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            word = merge_pair(bigram, list(word))
            word = tuple(word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        word = ' '.join(word)
        return word 
    
    def encode(self, text):
        tokens = []
        for token in re.findall(self.pat, text):
            token_bytes = [self.byte_encoder[b] for b in token.encode('utf-8')]
            bpe_tokens = self.tokenize(token_bytes).split(' ')
            tokens.extend(self.encoder[bpe_token] for bpe_token in bpe_tokens if bpe_token in self.encoder)
        return tokens

    def decode(self, token_ids):
        text = ''.join([self.decoder[id] for id in token_ids])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text

    def save_merges_vocab(self, merges, vocab):
        with op