import regex as re
import json
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab={}, merges={}):
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
            for word in self.pat.findall(text["text"]):
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
                pairs = self.get_pairs(word)
                for pair in pairs:
                    pair_freqs[pair] += freq
            if not pair_freqs:
                break

            # Most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Merge pair in all words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self.merge_pair(best_pair, list(word))
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

            # Add new token to vocab
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)
            merges[best_pair] = new_token
        self.save(vocab, merges)

    def tokenize(self, token):
        """
        Apply Byte Pair Encoding (BPE) merges to a single token.

        This function assumes the input token has already been converted to
        byte-level unicode symbols (via bytes_to_unicode). It repeatedly merges
        adjacent symbol pairs according to BPE merge ranks until no valid merge
        remains.

        Merges are applied in order of priority (lower rank = higher priority),
        exactly as in GPT-style byte-level BPE.

        Parameters
        ----------
        token : list[str] or tuple[str]
            A sequence of byte-level unicode symbols representing one pre-tokenized
            text chunk.

        Returns
        -------
        str
            A space-separated string of final BPE subword tokens.

        Examples
        --------
        Given BPE merges:
            {('h', 'e'): 0, ('l', 'l'): 1}

        Input token (after byte encoding):
            ['h', 'e', 'l', 'l', 'o']

        BPE steps:
            ['h','e','l','l','o'] -> merge ('h','e')
            ['he','l','l','o']   -> merge ('l','l')
            ['he','ll','o']      -> stop

        Output:
            "he ll o"
        """
        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            word = self.merge_pair(bigram, list(word))
            word = tuple(word)
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)

        word = ' '.join(word)
        return word


    def encode(self, text):
        """
        Encode a string into a sequence of integer token IDs using byte-level BPE.

        The encoding process consists of:
        1. Regex-based pre-tokenization (GPT-2 style).
        2. UTF-8 byte encoding of each token.
        3. Byte-to-unicode symbol mapping.
        4. BPE merge application.
        5. Mapping final BPE tokens to integer IDs.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        list[int]
            List of integer token IDs corresponding to BPE subwords.

        Examples
        --------
        Input:
            "hello"

        Pre-tokenization:
            ["hello"]

        Byte encoding:
            ['h','e','l','l','o']

        BPE output:
            ["he", "ll", "o"]

        Final token IDs:
            [4, 5, 3]
        """
        tokens = []
        for token in re.findall(self.pat, text):
            token_bytes = [self.byte_encoder[b] for b in token.encode('utf-8')]
            bpe_tokens = self.tokenize(token_bytes)
            tokens.extend(
                self.encoder[bpe_token]
                for bpe_token in bpe_tokens
                if bpe_token in self.encoder
            )
        return tokens


    def decode(self, token_ids):
        """
        Decode a sequence of integer token IDs back into a UTF-8 string.

        This reverses the encoding process:
        1. Token IDs -> BPE subword strings.
        2. Concatenate subwords.
        3. Unicode symbols -> raw bytes.
        4. UTF-8 decoding.

        Decoding is fully reversible for any text encoded with this tokenizer.

        Parameters
        ----------
        token_ids : list[int]
            List of integer token IDs.

        Returns
        -------
        str
            Decoded UTF-8 text.

        Examples
        --------
        Input token IDs:
            [4, 5, 3]

        Subwords:
            ["he", "ll", "o"]

        Output:
            "hello"
        """
        text = ''.join([self.decoder[id] for id in token_ids])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text

    def save(self, vocab,merges):
        with open("vocab.json", 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        with open("merges.txt", 'w', encoding='utf-8') as f:
            for merge in merges:
                f.write(" ".join(merge) + "\n")

    @classmethod
    def load(cls):
        with open("vocab.json", 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        with open("merges.txt", 'r', encoding='utf-8') as f:
            merges = [tuple(line.strip().split()) for line in f]
        
        return cls(vocab=vocab, merges=merges)