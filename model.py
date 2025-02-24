# codingium: disable


import math
import torch 
import torch.nn as nn





class LayerNorm(nn.Module):
  

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias




# x is the iput tensor of shape ( batch_size, seq_len))
class InputEmbedding(nn.Module):
    def __init__(self, d_model:int , vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)*math.sqrt(self.d_model)





 # x is the input tensor of shape (batch_size, seq_len, d_model)   
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x


class MultiheadAttention(nn.Module):
    # 
    def __init__(self, d_model: int,num_heads: int,dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.dk =d_model//num_heads 

        # be sure that d_model is divisible by num_heads
        assert d_model % num_heads == 0

        # define the linear layers for Q, K, V
        # wq is a linear layer  input dim :batch_size x seq_len x d_model output dim: batch_size x seq_len x d_model
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias = False)
        self.Wv = nn.Linear(d_model, d_model,   bias = False)

        # define the output linear layer
        self.Wo = nn.Linear(d_model, d_model, bias = False)

    @staticmethod

    def attention(Q, K, V, dk, mask, dropout: nn.Dropout):
        #compute the attention score 
        # Q, K, V are of shape (batch_size x num_heads x seq_len x dk)
        # Compute the score as the dot product==> Q*K^T/sqrt(dk) ==>(32*8*10*64) * (32*8*10*64)^T= (32*8*10*10)
        score = torch.matmul(Q, K.transpose(-2, -1)/math.sqrt(dk))


        # apply the mask to the score ( fill the masked values with -1e9)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)


        # apply the softmax function to the score
        score = torch.nn.functional.softmax(score, dim=-1)

        # apply the dropout to the score

        if dropout is not None:
            score = dropout(score)



        #compute the output of the attention layer
        # 32*8*10*10 * 32*8*10*64 = 32*8*10*64
        return torch.matmul(score, V), score 

    def forward(self, q, k, v, mask):
        # Compute Q, K, V ( batch_size x seq_len x d_model)
        # ex : x.size() = (32 x 10 x 512)
        K= self.Wk(k)
        Q= self.Wq(q)
        V= self.Wv(v)
        # For multihead attention, we split the d_model into num_heads(batch_size x seq_len x num_heads x dk)
        # ex: K.size() = (32 x 8 x 10 x 64)
        queries = Q.view(Q.size(0), Q.size(1), self.num_heads, self.dk).transpose(1, 2)
        keys =K.view(K.size(0), K.size(1), self.num_heads, self.dk).transpose(1, 2)
        values = V.view(V.size(0), V.size(1), self.num_heads, self.dk).transpose(1, 2)

        # compute the attention score
        x, self.attention= MultiheadAttention.attention(queries, keys, values, self.dk, mask ,self.dropout)

        # concatenate the heads
        att =x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)


        # apply the output linear layer
        return self.Wo(att)


class Residual(nn.Module):
    def __init__(self,d_model, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm= LayerNorm(d_model)



    def forward(self, x , sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(x))))




class EncoderBlock(nn.Module):
    def __init__(self, d_model : int , self_attention_block:MultiheadAttention, feed_forward_block : FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual = nn.ModuleList([Residual(d_model, dropout) for _ in range(2)])

    def forward(self, x , mask):
        x= self.residual[0](x, lambda x: self.self_attention_block(x,x, x, mask))
        x=self.residual[1](x, self.feed_forward_block)
        
        return x 


class Encoder(nn.Module):
    def __init__(self,  n_features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(n_features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
       

        return self.norm(x)




class DecoderBlock(nn.Module):
    def __init__ (self,d_model: int ,  self_attention_block: MultiheadAttention, cross_attention_block: MultiheadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual =nn.ModuleList([Residual(d_model,dropout) for _ in range(3)])


    def forward(self , x, encoder_output, src_mask, tgt_mask):
       x = self.residual[0](x, lambda x: self.self_attention_block(x,x,x , tgt_mask))
       x = self.residual[1](x,  lambda x: self.cross_attention_block(x,encoder_output, encoder_output, src_mask))
       x = self.residual[2](x, self.feed_forward_block)

       return x




class Decoder(nn.Module):
    def __init__(self,  n_features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(n_features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)



class ProjectionLayer(nn.Module):
    def __init__(self,    d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        # batch_size x seq_len x vocab_size ---> batch_size x seq_len x vocab_size
        return torch.log_softmax(self.proj(x), dim= -1) #torch.log_softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):
    def __init__ (self, encoder:Encoder, decoder:Decoder,src_emb: InputEmbedding,tgt_emb :InputEmbedding,src_pos : PositionalEncoding, tgt_pos : PositionalEncoding,  projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src=self.src_emb(src)
        src=self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output,  src_mask, tgt, tgt_mask):
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
     # create embedding 
    src_emb = InputEmbedding(d_model, src_vocab_size)
    tgt_emb = InputEmbedding(d_model, tgt_vocab_size)


    # create positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len)

    # create attention bloc 
    encoder_bocks=[]
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model ,encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_bocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model,decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_bocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer( d_model, tgt_vocab_size)


    # create tranformer model
    transformer = Transformer(encoder, decoder, src_emb, tgt_emb, src_pos, tgt_pos, projection_layer)

    # initialize the weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
