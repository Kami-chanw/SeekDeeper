[\[📖中文ReadMe\]](./README_zh.md)

## Introduction
Here, I implemented a transformer and used its encoder for text sentiment classification on the IMDB dataset (see [here](./train_imdb.ipynb)).

## Model details
### [Transformer](./modules/transformer.py)
Transformer was originally proposed to be used to solve translation tasks. If we want to translate from Chinese to English, we call Chinese the source language and English the target language. The structure of the Transformer is shown in the figure below. The embedding of the source text is added with positional encoding and then input into the encoder. After passing through N layers of encoder layers, the output interacts with the cross attention in the decoder. The embedding of the target text is similarly added with positional encoding and then input into the decoder. The output of the decoder usually goes through a linear layer (depending on the specific task).
<div style="text-align: center;">
  <img src="./images/transformer.png" alt="Transformer" style="width: 300px; height: auto;">
</div>

The encoder and decoder use two types of masks, `src_mask` and `tgt_mask`. The purpose of `src_mask` is to mask all PAD tokens to avoid their calculation in attention. `tgt_mask` not only masks all PAD tokens but also prevents the model from accessing future words when making next word predictions.

### [Positional Encoding](./modules/layers.py)
Unlike RNNs, the Transformer does not naturally have the characteristic of sequences, which leads to the loss of order information during the calculation of attention. Therefore, positional encoding needs to be introduced. In the original paper, the calculation formula for positional encoding is as follows:

- For even dimensions:
  $$
   \text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
  $$

- For odd dimensions:
  $$ 
  \text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) 
  $$

For numerical stability, we take the exponent and logarithm of the div term:
$$
\text{div\_term} = 10000^{2i/d_{\text{model}}} = \exp\left(\frac{2i\cdot-\log(10000)}{d_{model}}\right)
$$

The same sine and cosine positional encoding is used for any sequence, so the shape of the positional encoding is `[seq_len, d_model]`. It is then added to the input embedding of shape `[batch_size, seq_len, d_model]` using the broadcasting mechanism to get the encoder input, denoted as $x_0$.

## [Encoder](./modules/encoder.py)
The encoder consists of multiple identical layers. The output of the previous layer, $x_i$, passes through the following processes in the layer (omitting dropout):
```python
# attention mechanism
residual = x
x = multihead-attention(q=x, k=x, v=x, mask=src-mask)
x = layer-norm(x + residual)

# position-wise feed forward
residual = x
x = feed-forward(x)
x = layer-norm(x + residual)
```

## [Attention](./modules/layers.py)
The computation process of attention is as follows:
<div style="text-align: center;">
  <img src="./images/attention.png" alt="Attention" style="width: 400px; height: auto;">
</div>
In the encoder's self-attention, K, Q, and V are obtained from the previous layer's output through different linear layers. In the decoder's cross-attention, K and V come from the output of the encoder's last layer, while Q is from the previous layer of the decoder.

To make the model pay attention to different subspaces of different positions, we need to use multi-head attention. Specifically, K, Q, and V of shape `[batch_size, seq_len, d_model]` are split into `[batch_size, seq_len, n_head, d_key]`, and then the `seq_len` and `n_head` dimensions are swapped to facilitate matrix multiplication in the attention mechanism. After computing the attention, the results are merged and passed through a linear layer to map back to the original input dimensions. The algorithm flow is as follows:
```python
# projection
K, Q, V = W_k(x), W_q(x), W_v(x)

# split
d_key = d_model // n_head
K, Q, V = (K, Q, V).view(batch_size, seq_len, n_head, d_key).transpose(1, 2)
out = scaled_dot_product_attention(K, Q, V)

#concatenate
out = out.transpose(1, 2).view(batch_size, seq_len, d_model)
out = W_cat(out)
```

Scaled Dot-Product Attention is expressed by the formula:

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_{key}}}\right)\cdot V
$$

## [Decoder](./modules/decoder.py)
Compared to the encoder, the decoder has an additional cross-attention layer and uses masked multi-head attention, as the model is required not to interact with future information at this point, also known as causal self-attention.
The decoder also consists of multiple identical layers, where the output `enc` of the encoder's last layer and the output `dec` of the previous layer of the decoder pass through the following processes in the layer (omitting dropout):
```python
# causal self-attention
residual = dec
x = multihead-attention(q=dec, k=dec, v=dec, mask=tgt-mask)
x = layer-norm(x + residual)

# cross attention
x = multihead-attention(q=x, k=enc, v=enc, mask=tgt-mask)
x = layer-norm(x + residual)

# position-wise feed forward
residual = x
x = feed-forward(x)
x = layer-norm(x + residual)
```
