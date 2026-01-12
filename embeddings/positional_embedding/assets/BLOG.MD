# Visualizing Positional Encodings: What Actually Changes Inside a Transformer
Transformers are a type of neural network that are used for natural language processing (NLP). It came as a huge advancement to the field of NLp and became the go to model for NLP tasks like machine translation, text summarization, question answering, and more replacing the previous state of the art models like Recurrent neural networks and Long Short Term Memory networks. 

In previous models they are sequential models which means they process words one by one and memorie the order of the sequence, 
Transformers does this differntly, it processes words in parallel and compute attention for each word, this makes it permutation invariant, to the model the sequnece is just a cloud of data points.

If we dont tell the transfoer model where each word is located 
sentnce ike  The cat sat on the chair will be the same as The chair on sat the cat, or The man bit the dog and the cat bit the dog we can notice the incohersible manner of this sentnce and how much positon matters

!(A visualization of randomly sampled tokens without positonal encoding)[without_positons.gif]
Transformers fix this order by intorducing Postional Encoding to its arcitecture, this ensures order is added to the token before atteion is being computed.
In the transformer architecture relaesed by Viswwani et all in 2017 a paper titled Attention is all you need, they made use of Sinuisodal positional encoding which we willl reach there in a second. 

Ad over the u=years researchers in the field have found fault in each other particluarly with the problem of extrapolation which is how it performs on unseen data from its training, how it could scale in sequence length  invention or way of positional encoding and release their own  form of soultion, this has led to many techniques for doing this

But in this post i will ediscussing four of these with visuaization which are 

Sinuisodal
Learned Embeddings
RoPe rotary positonal embedding
Alibi Attention with linear biases

## Sinuisoal

Sinuodal like i said was introdyced in 2017 inthe original transfrmer paper as the go to way for positional encoding and there was a major reason behind choosig it, the team believed Sinuisodal would extrapolate to longer input sequences during infernce which wwas proven to be false 

The idea behind it was pretty simple, it follows a basic detreminsitic computation i.e
fror a given seq_length and d_modal compute the position embeddings for this i.e siuisodal_position_embeddings(seq_length, d_modal) using the formula 
For even positions:

 
P
E
p
o
s
,
2
i
=
sin
(
p
o
s
10000
2
i
/
d
m
o
d
e
l
)
 

For odd positions:

 
P
E
p
o
s
,
2
i
+
1
=
cos
(
p
o
s
10000
2
i
/
d
m
o
d
e
l
)

k
 : The position of the word in the sentence (for example, 0 for the first word, 1 for the second, and so on.)

 
i
 : The dimension index of the embedding vector. maps to column index. 2i will indicate an even position and 2i+1 will indicate an odd position

 
d
m
o
d
e
l
 : The predefined dimensionality of the token embeddings (for example, 512)

 
n
 : user-defined scaler value (for example, 10000)

 
P
E
 : position function for mapping position k in the input sequence to get the positional mapping



The raeson behind this approaxh is because sin and cosine are periodic function 
It is smooth and continuous: Small changes in input result in small changes in output, which gives us a way to represent positions in a differentiable space. and prevnts explosion of signals constraining thevalues between [-1, 1]

By varying the frequency of the wavelengths across dimensions, we can create a rich, multiscale representation of position. 
adding this computed detreminsitic values to our tokens remove the noise in our visualization as seen above

This we can see in the visualization below
!()[sinuisodal_visualization_3d.gif]

The values computed from the formul are added tp pur word tokens embedding contributing to it absolute nature this leads to additonal tasl for the mdoel to un mix the psoitin and from the semantic meaning.

and reseatch has shown relative encoding works best at psoitonal encoding compared to absolute
2
Learned embeddings

LErned embeddings was an alternative to the sinuisodal PE it was an approach released alongside sinuisoda; in the hope to LEARN position this makes it a stochatsic model differnt from how our sinuisodal model works.

The idea sounds pretty decent as we know a trained model will definitely performs better than a computed function in learning complex relationship, this was proven true as not only positions in sequences were learned during training but the model captures more semantic relationship like a word that appears at the begiining of a sequence has a relationship to the word thta appears at the end. Well this does sounds really great and will contribute to the performance of the model. yOu mighgt ask why is this not the go to approch to PE since it performs better than Sinuisodal in cpturing reltionship and was invented by the Viswani et al team.
The simple answer here is it couldn't extrapolate to a longer input sequnece at all
in the case of sinuisdoal given a L no of tokens trainning and k to be length of our extrapolated tokems  during inference for sinuisdoal k could extend to 50 and maintain it performance but Learned emebddings couldn't perform at even k = 1
This was a mojor setback fror it as in the word of ML we hope to prevent overfitting and our model is only good on hw it performs on unseen data during trainng

3 RoPE also known as Rotary positional Embedding    