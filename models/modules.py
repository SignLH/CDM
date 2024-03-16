import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Parameter,
    Linear,
    GELU,
    ReLU,
    LayerNorm,
    Dropout,
    Softplus,
    Embedding,
)
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch.nn.functional as F
import numpy as np
from enum import IntEnum


class MASKTTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super(MASKTTransformerLayer, self).__init__()
        """
            This is a Basic Block of Transformer paper.
            It contains one Multi-head attention object.
            Followed by layer norm and position-wise feed-forward net and dropotu layer.
        """
        kq_same = kq_same == 1

        self.masked_attn_head = MultiHeadAttentionWithIndividualFeatures(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.linear1 = Linear(d_model, d_ff)
        self.activation = GELU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block: object of type BasicBlock(nn.Module). It contains maksed_attn_head objects which is of type MultiHeadAttnetion(nn.Module).
            mask: 0 means that it can peek (엿보다) only past values. 1 means that block can peek only current and past values
            query: Queries. In Transformer paper it is the input for both encoder and decoder
            key: Keys. In transformer paper it is the input for both encoder and decoder
            values: Values. In transformer paper it is the input for encoder and encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the alyer andr returned
        """

        batch_size, seqlen = query.size(0), query.size(1)
        """
        when mask==1
        >>> nopeek_mask (for question encoder, knoweldge encoder)
            array([[[[0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]]]], dtype=uint8)

         >>> src_mask
            tensor([[[[ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]]]])

        when mask==0 (for knowledge retriever)
        >>> nopeek_mask
            array([[[[1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1]]]], dtype=uint8)

        >>> src_mask
            tensor([[[[False, False, False, False, False],
                    [ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False]]]])

        row: target, col: source
        """
        device = query.get_device()
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        bert_mask = torch.ones_like(src_mask).bool()

        if mask == 0:
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask)
        elif mask == 1:
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask)
        else:
            query2, attn = self.masked_attn_head(query, key, values, mask=bert_mask)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)

        return query, attn


class AKTTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super(AKTTransformerLayer, self).__init__()
        """
            This is a Basic Block of Transformer paper.
            It contains one Multi-head attention object.
            Followed by layer norm and position-wise feed-forward net and dropotu layer.
        """
        kq_same = kq_same == 1

        self.masked_attn_head = MultiHeadAttentionWithContextDistance(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.linear1 = Linear(d_model, d_ff)
        self.activation = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block: object of type BasicBlock(nn.Module). It contains maksed_attn_head objects which is of type MultiHeadAttnetion(nn.Module).
            mask: 0 means that it can peek (엿보다) only past values. 1 means that block can peek only current and past values
            query: Queries. In Transformer paper it is the input for both encoder and decoder
            key: Keys. In transformer paper it is the input for both encoder and decoder
            values: Values. In transformer paper it is the input for encoder and encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the alyer andr returned
        """

        batch_size, seqlen = query.size(0), query.size(1)
        """
        when mask==1
        >>> nopeek_mask (for question encoder, knoweldge encoder)
            array([[[[0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]]]], dtype=uint8)

         >>> src_mask
            tensor([[[[ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]]]])

        when mask==0 (for knowledge retriever)
        >>> nopeek_mask
            array([[[[1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1]]]], dtype=uint8)

        >>> src_mask
            tensor([[[[False, False, False, False, False],
                    [ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False]]]])

        As a result, the upper triangular elements are masked
        row: target, col: source
        """
        device = query.get_device()
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask)
        elif mask == 1:
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask)
        else:
            raise NotImplementedError

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)

        return query, attn


class MultiHeadAttentionWithIndividualFeatures(Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super(MultiHeadAttentionWithIndividualFeatures, self).__init__()
        """
        It has projection layer for getting keys, queries, and values. Followed by attention and a connected layer.
        """

        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = Linear(d_model, d_model, bias=bias)
        self.k_linear = Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = Linear(d_model, d_model, bias=bias)
        self.dropout = Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = Linear(d_model, d_model, bias=bias)
        self.gammas = Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        gammas = self.gammas
        scores, attn_scores = individual_attention(
            q, k, v, self.d_k, mask, self.dropout, gammas
        )

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output, attn_scores


class MultiHeadAttentionWithContextDistance(Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super(MultiHeadAttentionWithContextDistance, self).__init__()
        """
        It has projection layer for getting keys, queries, and values. Followed by attention and a connected layer.
        """

        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = Linear(d_model, d_model, bias=bias)
        self.k_linear = Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = Linear(d_model, d_model, bias=bias)
        self.dropout = Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = Linear(d_model, d_model, bias=bias)
        self.gammas = Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        gammas = self.gammas
        scores, attn = monotonic_attention(
            q, k, v, self.d_k, mask, self.dropout, gammas
        )

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output, attn


def individual_attention(q, k, v, d_k, mask, dropout, gamma=None):
    """
    This is called by MultiHeadAttention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float()

        distcum_scores = torch.cumsum(scores_, dim=-1)

        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

        device = distcum_scores.get_device()
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor)
        position_effect = position_effect.to(device)

        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    m = Softplus()

    gamma = -1.0 * m(gamma).unsqueeze(0)

    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )

    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    attn_scores = scores
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn_scores


def monotonic_attention(q, k, v, d_k, mask, dropout, gamma=None):
    """
    This is called by MultiHeadAttention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float()

        distcum_scores = torch.cumsum(scores_, dim=-1)

        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        """
        >>> x1-x2
            tensor([[ 0,  1,  2,  3,  4],
                    [-1,  0,  1,  2,  3],
                    [-2, -1,  0,  1,  2],
                    [-3, -2, -1,  0,  1],
                    [-4, -3, -2, -1,  0]])

        >>> torch.abs(x1-x2)
            tensor([[0, 1, 2, 3, 4],
                    [1, 0, 1, 2, 3],
                    [2, 1, 0, 1, 2],
                    [3, 2, 1, 0, 1],
                    [4, 3, 2, 1, 0]])
        """
        device = distcum_scores.get_device()
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor)
        position_effect = position_effect.to(device)

        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    m = Softplus()

    gamma = -1.0 * m(gamma).unsqueeze(0)

    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )

    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    attn = scores

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        self.postional_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.postional_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class CosinePositionalEmbedding(Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class BERTEmbeddings(nn.Module):
    def __init__(self, num_skills, hidden_size, seq_len, dropout, padding_idx=0):
        super(BERTEmbeddings, self).__init__()
        self.item_embeddings = Embedding(
            num_skills, hidden_size, padding_idx=padding_idx
        )
        self.positional_embeddings = Embedding(seq_len, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        item_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.positional_embeddings(position_ids)
        embeddings = item_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
