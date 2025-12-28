# Tokenization and Encoding with Byte Pair Encoding

Tokenization is an integral part of the field of Natural language processing(NLP) which is the frontier of how our modern llms came to be.
You see computer don't understand language like Human does, they have the ability to understand numbers and can calculate much more faster than human can.

Tokenization a building block for encoding is a preprocessing technique which helps with this, it allows us to represent language much more efficient and better so computer can understand.
<!-- in my previosu text you might notice i have been reffering to my letters in lowercase so that means 26 numbers to represent small alphabets another 26 to represnt 26 alphabets what about chinese characters? Japanese indian? do we map eachcharacters for each numbers  -->
Why Tokenization matters:
    since i said computers understand numbers far better than human can, a question might pop up in your mind why can't we just represent all alphabets by a number then encode our text corresponding to that number for instance all occurence of "a" will be 1 and "z" will be 26, okay easy there genius well this in simple terms is what encoding in NLP means representing tokens or as you might refer to it words(well it is not just word we will get there in a moment ) in numbers let's say a is 1 b is 2 and so on we can represent hello in numbers as 8 5 12 12 15, computer will surely understand this but for every solution we come up with there is always a room for an efficiency problem, this means we represnt the numbers of word is represented in the same length as our word hello is 5 length same with 8 5 12 12 15, this is not efficient and lead to wastage of spaces and computer resources becuase my friend for your llm to be or any language based model you use trained on a neural network it passed through operations and represntationl, so our genius, you came up with another idea which is representing word by a number now this does seem efficient at leats now we can represnt a word like hello as 1 and world as  3 so our encoding represntation will be 1 3 for helloworld don't see this a typo, yes helloworld is differnet from hello world " " in terms of NLP is also a token so we have to assign a number for it too hello world might be somthing like 1 2 3, this does seem practical we just have to assign id to every word in the dictionary, wait there are over 500 000 english words alone, well we have to also consider morphology run can be running runner , what about other languages? oh there is emojis, symbols too slangs too this will make our represnattaion in millions of tokens to be represented, well oh genius this doesn't seem effective to me. This what gave birth to the some current tokenization and encoding techniques today, most of them started out as a compression technique to fully represent language.

Overview of Tokenization Technique
    Liek i said erlier what you invented is infact a tokenization technique reprsenting separting words into letters is a form of tokenization called Character level which we both know how it relates to efficiencyand with this it does as its strngth which is it doesn't encounter OOV (Out of vocabulary) problem which is simply not having a number to represnt a token since at foundation level all languages has a character and we can simply assign this charcters a number this means all words can easily be encoded 

    word level tokenization which i discussed earlier does offer a better way of compression to Charcter level tokenization but fails with OOV, due to the fact we can't account for all possible words.

    This is what modern Tokenizationn technique called sub word- level tokenization tries to fix, we have a bunch Unigram, WordPiece, BPE etc.
    i won't be going much into the details of other tcehniques aside BPE but they all strive to acheive maximum compression and effective represeatation, remember i mentioned something regarding tokens not being words alone they can be charcters or subwords for instance token can be [hel, eat, !, @, g, ing, ed, 1 , 2] these are all tokens and this is what the subword level techniques try to achieve it allows for represntation of words better like hello can become he ll o, this depends on what kind of tokens we have "he" as a unique id same as ll and o and for all occurence of he we represnt them with its id. Voila we just solve efficiency issue, we can fully represent all words even one as complex as "Donaudampfschiffahrtselektrizitätenhauptbetriebswerkbauunterbeamtengesellschaft" this is a german compounded word.
    now with less than 60000 tokens we can represnt all languages words.  Sound like a magic right? i know .

Why Byte-Level BPE (GPT-style)
    Byte Pair Encoding was actually introduced by Gage in 1994 as a form of compression algorithm, it is quite a simple technique yet the most effctive technique to grab, it is the frontier behind tokenization concept use for the preprocessing of data for training of modern Large Language Models or in other major NLP tasks.
    two major optimization problem Given a string s and an integer k > 0,
    find a merge sequence R of length k, of maximal utility for s (or equivalently, of minimal compressed
    length). We denote this optimal utility as OPTm(s, k), and call the task of computing it the optimal
    merge sequence (OMS) problem The optimal pair encoding (OPE) problem asks, given a string s and an integer k > 0,
    to find a partial merge sequence R
    ∗ of length k, of maximal utility for s. We denote this optimal
    utility as OPT(s, k).
    Byte-pair encoding (BPE). BPE solves both the OPE and OMS problem as follows. Starting
    with the input string s, it performs k locally optimal full merge steps, always choosing a pair whose
    replacement maximizes compression utility.
    Formally, for input (s, k), we output R = (R1, . . . , Rk), where Ri = replaceaibi→ci
    . Denoting
    s
    (0) = s, and s
    (i) = Ri(s
    (i−1)) for i ∈ [k], each ci
    is a new symbol, i.e., not occurring in s
    (j) with
    j < i, and for i = 1, . . . , k, the pair aibi
    is chosen so that |Ri(s
    (i−1))| is minimal.
    With careful data structuring, identifying aibi and performing Ri can be done in linear total time
    over all k merge steps, e.g., see [SKF+99]. In this paper, we ignore such implementation details
    and focus on the total utility of BPE, i.e., |s| − |s
    ′
    |, where s
    ′ = s
    (k)
    . We denote this quantity as
    BPE(s, k). Note that clearly BPE(s, k) ≤ OPTm(s, k) ≤ OPT(s, k).

    Although we won't dive in to much into optimization problems here rather how it relates to llm, it solves the OOV problem and also handles emojis, accents, rare scripts which was a major issue concerning tokenization simply by borrowing of the concept of `every string is representable as  a byte` this simple line changes everything, how? you might ask a chines chracter can be represnted as a byte.
    Before byte-level encoding, tokenizers worked on characters. This sounds fine for English, but the world has:

    Over 140,000 different Unicode characters (including emojis like 🚀 and complex scripts like Chinese or Hindi).

    If your "base" vocabulary has to include every single possible Unicode character just to avoid the <UNK> the unknown token, your vocabulary is already "exploded" before you even start merging pairs!
    By falling back to Bytes (UTF-8 encoding), we simplify the entire universe of text into just 256 possible base values.

    How this changes everything:
    The Universal Base: Instead of having a "base" vocabulary of 140,000+ characters, the "base" is just 256. This is tiny and manageable.

    No more <UNK>: Since every possible piece of digital text (from a simple 'a' to a complex 🇨🇳 flag) is ultimately just a sequence of bytes between 0 and 255, the model can always represent the input.

    Efficiency: BPE then starts merging these bytes. Common English letters might merge quickly, while a complex emoji might stay as a sequence of 3 or 4 byte-tokens, a chinese charcters canbe represented a more than 2 bytes.

BPE ALGORITHM:
    the first part to building a BPE is by pretokenization using regex as it was done in the GPT-2 BPE implementation they used r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" this ensures Handling contractions, punctuation, spaces a simple Hello world! text is seprated into "hello world!" → ["hello", " world", "!"]
    
    As we mentioned about the unicode byte represntation with 256 base bytes we can now represnt any charcters in our word symbols or chinesse charcters can be represnted by two letters this allows the conversion into charcters constricted by the 256 base bytes and with this utf-8 we can easiy reverse the charcter back to it origina symbol, this is such an important task in out BPE alogorithm
    here is a code that shows the byte to unicode representation
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
    and before traininng the intial vocab size is 256 excluding special tokens which are are all the possible utf-8 encoded charcters
    the training loop is such a simple thing to grasp it is a popularity contest algorithm which favors the most frequently occuring pairs 
    if we have a train corpus 
    corpus = [
    {"text": "low lower"},
    {"text": "low lowest"}
]

we extract words low, lower, low, lowest
calculate eacg word freq {
 ('l','o','w'): 2,
 ('l','o','w','e','r'): 1,
 ('l','o','w','e','s','t'): 1
}
 fr each word we extract pirs something like
 Pairs per word

('l','o','w') ×2

(l,o), (o,w) → each count +=2


('l','o','w','e','r')

(l,o), (o,w), (w,e), (e,r)


('l','o','w','e','s','t')

(l,o), (o,w), (w,e), (e,s), (s,t)

Total pair counts
(l,o): 4
(o,w): 4
(w,e): 2
(e,r): 1
(e,s): 1
(s,t): 1
theb we select the best pair and add it to our vocab lo and also crreate  amerge dictionary to track the merges so during next iteration the  word low will be represnted as ('lo', 'w')
this is done till we reach our vocab_size gpt-2 tokenizer have over 50000 vocab size trained over large datsets this ensure it can capture more better realtionship nd the frequency is more favorable

Now you just built the BPE train algorithm

tokenize porcess follows the same process with our train algorith, but now we have a vocab builded on ranking we can query when we compute pairs we can query the vocab_rank to check if the pair exist then we proceed to merging the  pair if it exist

encoding is just using our byte_encoder to encode the text with utf-8 and decode with the 256 base bytes this prevents OOV and then we tokenize then we can query the tokenized tokens in our vocab and get the encoded ids this built strictly by ranks
please notice  the use of strictly it really matters 
and decoding follows the same process just in reverse order

And yeah my friend you just built a BPETokenizer.

it has some limitations especially with the frequency computation and like i said it is a populrity contest algotihm but the sky is wide you can invent better algrithm  my friend

LLMs can't spell:
    as an addition the major reason behind the famous question llm couldn't answer is how many r's in strawberry this is mainly due to tokeniation how llm see the text strawberry if you check https://tiktokenizer.vercel.app/?model=gpt2 you can see the token starwberry is broken down into three different token 301, 1831, 8396 st raw berry this is how this is fed into the llm so the llm doesn't see charcters but token

Tokenization defines symbols.
Embeddings define meaning
We will look at embedding in next blog post 
thanks.