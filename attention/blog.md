# Why Attention Beats Recurrence: A Deep Dive from LSTM to Transformer
Introduction: Sequential Modeling in NLP

The field of NLP has experienced a series of evolution and major breakthroughs, the most visble of this and the hottest topic right now is the Large language models which can perform series of mnay tasks ranging from your text translation to question answering task to sentiment analysis you name it. We can call it a knowledge base which understand languages so well but this wasn't just something that came to be in 2017, infact this is a result of years of research dated to 1950s but didn't really take off due to limitation of Computing power(Note this) Major breakkthroughs came inn the 90s when we could compute statistical algorithm and it yield results. 

You see a Machine learning and statistics go hand in hand, most statitsical algoirthms used for the modeling were statistics that has been discovered 200 years ago, but with the advancement in compute power we were able to perform heavy computation using this algourthm for prrediction.

Deep neural network was part of this statsistical models and it really took off in the community of researchers then, especially with the introduction of backpropagation populrised by , David E. Rumelhart et al. but you see the idea of backpropagation was based on chain rule deerived by Gottfried Wilhelm Leibniz in 1673,Artificial Neural network follow same format given an input x, a model to be f(x) we try to predict y i.e y = f(x)
There were many innnovations and architecturees derived on this and all of it depends on how you model f(x) and it is the powerbone behind the whole industry of Deep learning right now, RNN(recursive neural network) was part of the innovatin that stems from it, unlike feedforward neural network which operates by y = f(x) on a single input, RNN borrow from the idea of recurrence and change the format of the equation to yt​=f(xt​,ht−1​) where xt is the input from a previous step, this simple idea here is made possible by shortcut connection a vital part of our rnn architecture where input from a previous time step is fed back as an input to the ext time step This enables RNNs to capture temporal dependencies and patterns within sequences.
It was successfullly applied in sequences tasks like connected handwriting recognition, speech recognition, natural language processing, and neural machine translation.
This was the foundation for our sequnetil modeling histoty

Traditional RNN suffers from  vanishing gradient problem, which limits their ability to learn long-range dependencies. this was fixed by variants architectures like LSTM(Long short term memory) and GRU(Gated Recurrent Units), LSTM introduced in the 90s became the go to architecture for many transduction model then being the backbone of seq2seq engine like google translate won the 2015 imagenet islvrc competition with the introduction of resnet Residual neural network ehich made use of LSTM as its memory mechanism and many other application in the indutstry. While this revolutionised sequential modeling, LSTM still has a major flaw it process text in sequence and couldn't manage long text window which renders it useless understanding meanningful point in a text and with moore law computation power increased drastically over the years, with the popularization of parallel computing and enough compute power, one might really ask the question if LSTM is really the best approach to Sequential modelling.

Attention was a popular concept or rather i say notion in Machine learning which involves parts in sequence realting to each other and have been around for some time, this would prove to be helpful in modeling long range sequences and the word at the beginning could realate with the word aat the end fo instance Micheal Jordan is a great basketballer, He really is the ______ unlike in RNNs where the idea of recent informtion in sequnce are favored fro making prediction attention added a notion to relating information at the beggining of the sequence to model its preiction. With the release of Attention is all you need paper in 2017 where they formalized the scaled dot product attention matrix and among many other layers of their transformer  architecture which supports parallel computing, it really became the go to approach in all AI/ML fields computer vision NLP you name it.
In this blog we will be looking at LSTM a variant architecture  of RNN and Transformer an architecture inspired by attention.

# LSTM architecture

LSTM meant to fix the long range deendency issues of RNN, one might ask how? RNN has repeating module in its layers 
LSTM handles recurrnce or lets say passing an information input from a previous state 
throught a special gating mechanism which controls informatin flows
The Four Gates:
🚪 Input Gate (i): Controls how much new information to let in

sigmoid(...) → values between 0 and 1
0 = "block everything", 1 = "let everything through"
🗑️ Forget Gate (f): Controls what to discard from memory

sigmoid(...) → values between 0 and 1
0 = "forget completely", 1 = "remember everything"
📝 Cell Gate (g): Creates new candidate information

tanh(...) → values between -1 and 1
The actual content to potentially add to memory
📤 Output Gate (o): Controls what to output from the cell

sigmoid(...) → values between 0 and 1
Filters what part of the cell state becomes the hidden state


State Updates:
c_next = f * c + i * g
Old memory (c) × forget gate (f) → what to keep
New info (g) × input gate (i) → what to add
Result: Updated cell state
h_next = o * torch.tanh(c_next)
Apply tanh to cell state → normalize to [-1, 1]
Multiply by output gate (o) → filter what to expose
Result: Hidden state (what the network "sees")

in the training sample on the github repo we build a character level language model with LSTM and it follows the principle of the gating mechanism and we could trace one training step with sequence "hello" from our training code :

Time:    t=0      t=1      t=2      t=3      t=4
Input:    h        e        l        l        o
Target:   e        l        l        o        ?
Initial state: h₀=[0,0,0...], c₀=[0,0,0...]
Step 1: Process 'h'
  x = embed['h']
  logits₁, (h₁, c₁) = LSTM(x, h₀, c₀)
  loss += CE(logits₁, 'e')
Step 2: Process 'e'  
  x = embed['e']
  logits₂, (h₂, c₂) = LSTM(x, h₁, c₁)  ← Uses previous state!
  loss += CE(logits₂, 'l')
Step 3: Process 'l'
  x = embed['l']
  logits₃, (h₃, c₃) = LSTM(x, h₂, c₂)  ← Context from 'h' + 'e'
  loss += CE(logits₃, 'l')
... and so on
Key insight: Each step uses the state from the previous step, allowing the model to "remember" earlier characters!

And when you stack it in multiples your models benefitted from learning more patterns in your sequence 

Here is an image that shows it

!(a clean, educational diagram showing LSTM architecture with these components:

Title: "LSTM Character-Level Language Model"

Show a flowchart with these layers from bottom to top:
1. Input: Character indices (shown as numbers like "5, 12, 8")
2. Embedding Layer: Converting indices to dense vectors (show as colored rectangles)
3. LSTM Layer 1: Show the 4 gates (Input gate, Forget gate, Cell gate, Output gate) with arrows showing information flow. Label hidden state h₁ and cell state c₁
4. LSTM Layer 2: Similar structure, with h₂ and c₂ states
5. Output Layer: Projects to vocabulary size, shows probability distribution
6. Output: Predicted next character

On the side, show:
- State flow: Arrows showing how h and c flow through time steps
- Gates visualization: Small boxes labeled i, f, g, o with sigmoid/tanh activations
- Color coding: Use blue for hidden states, green for cell states, orange for input/output

Include these annotations:
- "4 Gates control information flow"
- "Cell state: long-term memory"
- "Hidden state: short-term output"
- "Stacked layers learn hierarchical patterns")[link.img]
with this gating mechansism we could see why it was more powerful than the traditional RNN


# Transformer architecture and self attention
self attention fixes relation, it propose the fact that given a sequence we can compute the score at which each words n the sequence relate with each other, this disregard the notion of using recent information to predcit the next token but rather a token at the beginning of our sequence contributes to the prediction of the next token.

This is the fundamental aspect of Transformer the modern day architecture for sequntial modeling.
in the attention is all you need paper they formalize the notion of scaled dot attention matrix which is Attention(Q, K, V ) = softmax(QKT
√
dk
)V

!()[transformer.img]
where Q is the Query, K is the key and V is the value 
the idea is for each token in our sequence we compute a query vector key vector and value vector we compute the attention weight of how this token relate with each other, this is done by dot matrix product of the query and key vectors divided by the square root of the dimenison size, in an auto regressive task where we generte the next token we don't want cheating in our model training i.e we want to limit what the model can see at a time in a sequence during training so a token in a seqeunece only relates to itself and tokens before it. this is done by the Masking technique where upper triangle of our dot product matrix is filled and the lower part are replaced by -inf this ensures during softmax the sum is 1 and the lower tringle matrix are 0s and then we dot product wit value V
Note Cross atention a variation of self attention do nt make use of msking, masking ensures later tokens do not influece precceding tokens
This is self attention in our transformer we mkae use of multi head attention which is multiple attention heads performing same similar score operation but with the benefit of powerful computation this could be run in parallel making usre our model learns important details in how token relate to each other bsically just like an human learning about grammatical notions of  a language   and for the multi hhead attention it is really nothing fancy just size manipulation


there are certain layers in our transformer mostly to help with stabilizing training =, these are LayerNOrmlization which is the LayerNorm(x)i​=γi​⋅σ2+ϵ
​xi​−μ​+βi​al
It helps the keeps the network numerically sane so attention and FFNs can actually learn.

and there is also the Feed Forward layer, this is really something we have entioned earlier liek we said a FFN doesn't conncern itself with the environment it focuses on transforming input to values so the computatain done here doesn't influence the attention between tokens, it comes fater attention and focuses on represnting individual token in tehir own represntation Turns “raw attended context” into usable features

Enables abstraction, pattern extraction, and decision-ready representations it is where most of themodel paramters live to say 

Well in simple terms this is Transformer in autshelll

# How it performs against LSTM

We have a code repo showing the deonstartion of both for a charcter language model, thugh in the modern world we make use of a byete pair tokenizer for tokenization not chracter, this was just for the sake of an experiment, it is worth noting Trasformer learns quicly than LSTm it could be able to understand how chrcters realte with each other in forming words but struggle with positions though LSTM struggle for this a bit and learns gradually and could generate a correhesible snetence at a certain epoch after training this was partly due to datastet size, mode of tokenizationa nd training epochs.

It is worth notting when we mention how attention is compute and the wheole architecture of trsnforner we didn't mention somethings about order despite the fact it is ran parallel so how does the model learns to differntiate between The cat is sitting on the chair from Teh chair is sitting on the cat. Thsi si done with Postional embedding you can check my previous blog on this {link to my previous blog}.

And with the benefit of parallel computation we can generate better results in the long run and way better long range dependnecy compared to LSTM.

Also it is worth noting the fact of how Important masking is for this autoregressive generation the model overfit on training text just like it memorize it but performs really bad on unseen data. Thats why the idea of maksing helps.

# Sampling
When it comes to generating text you have to know certain things which are basically temperature top k and top n these are parameters passed to prevent a greedy approcah to model infernce, our input is passed through a liner layer which outputs a vectors of probs logits this correspond to the lieklihood of the token to be chosen without temperature, top_k, top_n our model uses argmax which justreturn the highest prob of the token so fo each generation the model produces similar output, to introduce he set of diversity to our model the oncept of temperature was introduces this reduces power for the highest prob token and tone the logits down then we use torch.multinomial to do random selection this in simple terms according to torch website is for a tesor [ 0.3, 0.6, 0.1, 0.1, 0.1] -> [a, b, c, d, e] torch.multinomial ensures b has 60% chance of being chosen and a has a 30% chance of being chosen this introduces diversity and creativity in our model output.top_k basically follows the concept of giving a vaue of k let's say 10 it picks the top 10 probs token then renormalized their probablity, choose form this 
Top n (nucleus samling) given n choose a small set of tokens probs that their cumulative sum equal to n then sample from this.

While this does intrdouces creativity to our model generation it poised a greater risks which is highe values of these could lead the model picking tokens which doesn't correlate to the context of the sequence at least lower probs are lower for a reason.

SO you have to be craeful with your approach an descision in choosing this.



# IN modern context 

Although the idea about sequential modeling especialy tranduction models which re the encder decoder models has beenn to acheive longer context rnge dependnecy i.e longer text window to process Transformer beat LSTM and this is what lead to ho we were able to create Question answering bot as  long as we could fit teh whole texts for the model to process in the same context window, and with the help of some Supervised Fien tuning(SFTs) the model could produce a menaingful response to teh user, modern tcehiauqes have spiraled out to address the bottleneck of the transformer by fixing it positional embedding as introduced by the authors several reseachers have cme up with a btter way to do this and it will increaese context window over infernce not only this certan part of our architecture has beeen changed to improve stabilty and training perfomanace in our traditional transformer we made use of GeLu actvation layer modern transformer are using RELu, LAyerNorm has been replaced with RmSLayerNorm more parameters so the model could earn more and more patterns, using of Kv Cache for fatser inference this poses a question IS more parameters, better positional embedding to improve the context widnow the best approch to acheiveing Artificial GEneral Intelligence?

Thanks