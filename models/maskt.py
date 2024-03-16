from abc import ABC
import math
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import dgl
from dgl.nn.pytorch.conv import GATConv

from torch.nn import (
    Module,
    Embedding,
    Linear,
    ReLU,
    Dropout,
    ModuleList,
    Softplus,
    Sequential,
    Sigmoid,
    BCEWithLogitsLoss,
)
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from .modules import MASKTTransformerLayer

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, args):
        super(MultilayerPerceptron, self).__init__()
        self.num_features = 10
        self.nhid = 256
        self._size = 256
        self.dropout_ratio = 0.01

        self.lin1 = torch.nn.Linear(self.num_features, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, 1)

    def size(a, b):
        return 256

    def get_device(a):
        return "cuda"

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        features = x

        x = torch.flatten(self.lin3(x))
        return x, features


class attnVec_dot(nn.Module, ABC):
    def __init__(self, args, num_mataPath, device):
        super(attnVec_dot, self).__init__()
        self.num_path = num_mataPath
        self.attnVec = nn.Parameter(
            torch.rand(size=(1, args.emb_dim, 1), device=device), True
        )

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        path_weight = F.softmax(torch.matmul(semantic_embeddings, self.attnVec), dim=1)

        ques_embedding = torch.sum(
            semantic_embeddings * path_weight, dim=1, keepdim=False
        )
        return ques_embedding, path_weight


class attnVec_nonLinear(nn.Module, ABC):
    def __init__(self, args, num_metaPath, device):
        super(attnVec_nonLinear, self).__init__()
        self.num_path = num_metaPath
        self.attnVec = nn.Parameter(
            torch.rand(size=(1, args.emb_dim, 1), device=device), True
        )
        self.fc = nn.Linear(args.emb_dim, args.emb_dim).to(device)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        trans_embeddings = torch.tanh(self.fc(semantic_embeddings))
        path_weight = F.softmax(torch.matmul(trans_embeddings, self.attnVec), dim=1)

        ques_embedding = torch.sum(
            semantic_embeddings * path_weight, dim=1, keepdim=False
        )
        return ques_embedding, path_weight


class attnVec_topK(nn.Module, ABC):
    def __init__(self, args, num_metaPath, device):
        super(attnVec_topK, self).__init__()
        self.num_path = num_metaPath
        self.top_k = args.top_k
        self.emb_dim = args.emb_dim
        self.attnVec = nn.Parameter(
            torch.rand(size=(1, args.emb_dim, 1), device=device), True
        )
        self.fc = nn.Linear(args.emb_dim, args.emb_dim).to(device)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        path_weight = torch.matmul(
            torch.tanh(self.fc(semantic_embeddings)), self.attnVec
        )

        select_weight = torch.topk(path_weight, k=self.top_k, dim=1, sorted=False)
        path_weight = select_weight.values
        path_weight = F.softmax(path_weight, dim=1)
        index = select_weight.indices.repeat(1, 1, self.emb_dim)
        ques_embeddings = torch.gather(semantic_embeddings, dim=1, index=index)

        ques_embedding = torch.sum(ques_embeddings * path_weight, dim=1, keepdim=False)
        return ques_embedding, path_weight


class HetGAT_Emb(nn.Module, ABC):
    def __init__(self, args, device):
        super(HetGAT_Emb, self).__init__()
        self.device = device
        self.edge_list, self.g_list, self.feat_list = [], [], []
        self.num_ques = args.num_ques
        self.metaPaths = args.meta_paths
        self.fusion = args.fusion
        self.num_metaPath = len(self.metaPaths)

        self.gen_edges_feats(args)

        self.het_gat = HetGAT(
            self.g_list, self.feat_list, args.emb_dim, self.num_metaPath, args.gat_heads
        )

        self.semantic_attention = self.get_semantic_attention(args)

        self.element_weights = None
        self.semantic_weight = None

    def forward(self, pad_ques):
        emb_list, self.element_weights = self.het_gat()

        ques_emb_list = [emb[: self.num_ques, :] for emb in emb_list]
        ques_emb, self.semantic_weight = self.semantic_attention(ques_emb_list)
        batch_ques_emb = F.embedding(pad_ques, ques_emb)
        return batch_ques_emb

    def gen_edges_feats(self, args):

        entity2emb_dict, entity2num_dict = {}, {}
        type2name_dict = {
            "q": "question",
            "u": "train_user",
            "k": "skill",
            "t": "template",
        }
        type_str = "_".join(args.meta_paths)
        for t in ["q", "u", "k", "t"]:
            if t in type_str:
                entity2num_dict[t] = len(
                    eval(
                        open(
                            os.path.join(
                                args.data_path,
                                args.data_set,
                                "encode",
                                "%s_id_dict.txt" % type2name_dict[t],
                            )
                        ).read()
                    )
                )
                entity2emb_dict[t] = nn.Parameter(
                    torch.randn(entity2num_dict[t], args.emb_dim), requires_grad=True
                )

        for mp in self.metaPaths:
            feature = torch.cat(
                [entity2emb_dict[mp[0]], entity2emb_dict[mp[1]]], dim=0
            ).to(self.device)
            self.feat_list.append(feature)

            edge = np.load(
                "%s/%s/adj_mat/%s_Edge.npy" % (args.data_path, args.data_set, mp)
            )
            print("num edge of different meta-path, %s: %s" % (mp, edge.shape[1]))
            miss_node_set = set(
                range(entity2num_dict[mp[0]] + entity2num_dict[mp[1]])
            ) - set(edge.flatten())
            miss_node_np = np.array(list(miss_node_set), dtype=int)
            g = dgl.graph(
                (
                    np.concatenate([edge[0], miss_node_np], axis=-1),
                    np.concatenate([edge[1], miss_node_np], axis=-1),
                )
            ).to(self.device)
            self.edge_list.append(torch.stack([g.edges()[0], g.edges()[1]], dim=0))
            self.g_list.append(g)

    def get_semantic_attention(self, args):
        assert self.fusion in ["attnVec_dot", "attnVec_nonLinear", "attnVec_topK"]
        if self.fusion == "attnVec_dot":
            return attnVec_dot(args, self.num_metaPath, self.device)
        elif self.fusion == "attnVec_nonLinear":
            return attnVec_nonLinear(args, self.num_metaPath, self.device)
        else:
            return attnVec_topK(args, self.num_metaPath, self.device)


class HetGAT(nn.Module, ABC):
    def __init__(self, gs, fs, emb_dim, num_path, heads_list):
        super(HetGAT, self).__init__()
        self.num_metaPath = num_path
        self.gs = gs
        self.fs = fs

        self.gat_list = nn.ModuleList()
        for i in range(self.num_metaPath):
            self.gat_list.append(
                GAT(self.gs[i], self.fs[i], emb_dim, len(heads_list[i]), heads_list[i])
            )

    def forward(self):
        semantic_embeddings, element_weights = [], []
        for i in range(self.num_metaPath):
            emb, wgt = self.gat_list[i]()
            semantic_embeddings.append(emb)
            element_weights.append(wgt)

        return semantic_embeddings, element_weights


class GAT(nn.Module, ABC):
    def __init__(self, g, f, in_dim, num_layers, heads):
        super(GAT, self).__init__()
        self.g = g
        self.f = f
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    in_dim,
                    in_dim // heads[i],
                    heads[i],
                    feat_drop=0.2,
                    attn_drop=0.2,
                    residual=True,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )

    def forward(self):
        attn_wgt = None
        h = self.f
        for i in range(self.num_layers):
            h, attn_wgt = self.gat_layers[i](self.g, h, get_attention=True)
            h = h.flatten(1)
        return h, attn_wgt


class MASKT(Module):
    def __init__(self, num_skills, num_questions, seq_len, **kwargs):
        super(MASKT, self).__init__()

        self.mlp = MultilayerPerceptron
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.seq_len = seq_len
        self.args = kwargs
        self.hidden_size = self.args["hidden_size"]
        self.num_blocks = self.args["num_blocks"]
        self.num_attn_heads = self.args["num_attn_heads"]
        self.kq_same = self.args["kq_same"]
        self.final_fc_dim = self.args["final_fc_dim"]
        self.d_ff = self.args["d_ff"]
        self.l2 = self.args["l2"]
        self.dropout = self.args["dropout"]
        self.reg_cl = self.args["reg_cl"]
        self.negative_prob = self.args["negative_prob"]
        self.hard_negative_weight = self.args["hard_negative_weight"]

        device = torch.device("cpu")

        self.question_embed = Embedding(
            self.num_skills + 2, self.hidden_size, padding_idx=0
        )
        self.interaction_embed = Embedding(
            2 * (self.num_skills + 2), self.hidden_size, padding_idx=0
        )
        self.sim = Similarity(temp=self.args["temp"])

        self.question_encoder = ModuleList(
            [
                MASKTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.interaction_encoder = ModuleList(
            [
                MASKTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.knoweldge_retriever = ModuleList(
            [
                MASKTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out = Sequential(
            Linear(2 * self.hidden_size, self.final_fc_dim),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )

        self.cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, batch):

        if self.training:
            q_i, q_j, q = batch["skills"]
            r_i, r_j, r, neg_r = batch["responses"]
            attention_mask_i, attention_mask_j, attention_mask = batch["attention_mask"]

            ques_i_embed = self.question_embed(q_i)
            ques_j_embed = self.question_embed(q_j)
            inter_i_embed = self.get_interaction_embed(q_i, r_i)
            inter_j_embed = self.get_interaction_embed(q_j, r_j)

            if self.negative_prob > 0:

                inter_k_embed = self.get_interaction_embed(q, neg_r)

            ques_i_score, ques_j_score = ques_i_embed, ques_j_embed
            inter_i_score, inter_j_score = inter_i_embed, inter_j_embed

            for block in self.question_encoder:
                ques_i_score, _ = block(
                    mask=2,
                    query=ques_i_score,
                    key=ques_i_score,
                    values=ques_i_score,
                    apply_pos=False,
                )
                ques_j_score, _ = block(
                    mask=2,
                    query=ques_j_score,
                    key=ques_j_score,
                    values=ques_j_score,
                    apply_pos=False,
                )

            for block in self.interaction_encoder:
                inter_i_score, _ = block(
                    mask=2,
                    query=inter_i_score,
                    key=inter_i_score,
                    values=inter_i_score,
                    apply_pos=False,
                )
                inter_j_score, _ = block(
                    mask=2,
                    query=inter_j_score,
                    key=inter_j_score,
                    values=inter_j_score,
                    apply_pos=False,
                )
                if self.negative_prob > 0:
                    inter_k_score, _ = block(
                        mask=2,
                        query=inter_k_embed,
                        key=inter_k_embed,
                        values=inter_k_embed,
                        apply_pos=False,
                    )

            pooled_ques_i_score = (ques_i_score * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1)
            pooled_ques_j_score = (ques_j_score * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)

            ques_cos_sim = self.sim(
                pooled_ques_i_score.unsqueeze(1), pooled_ques_j_score.unsqueeze(0)
            )

            ques_labels = torch.arange(ques_cos_sim.size(0)).long().to(q_i.device)
            question_cl_loss = self.cl_loss_fn(ques_cos_sim, ques_labels)

            pooled_inter_i_score = (inter_i_score * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1)
            pooled_inter_j_score = (inter_j_score * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)

            inter_cos_sim = self.sim(
                pooled_inter_i_score.unsqueeze(1), pooled_inter_j_score.unsqueeze(0)
            )

            if self.negative_prob > 0:
                pooled_inter_k_score = (
                    inter_k_score * attention_mask.unsqueeze(-1)
                ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                neg_inter_cos_sim = self.sim(
                    pooled_inter_i_score.unsqueeze(1), pooled_inter_k_score.unsqueeze(0)
                )
                inter_cos_sim = torch.cat([inter_cos_sim, neg_inter_cos_sim], 1)

            inter_labels = torch.arange(inter_cos_sim.size(0)).long().to(q_i.device)

            if self.negative_prob > 0:
                weights = torch.tensor(
                    [
                        [0.0] * (inter_cos_sim.size(-1) - neg_inter_cos_sim.size(-1))
                        + [0.0] * i
                        + [self.hard_negative_weight]
                        + [0.0] * (neg_inter_cos_sim.size(-1) - i - 1)
                        for i in range(neg_inter_cos_sim.size(-1))
                    ]
                ).to(q_i.device)
                inter_cos_sim = inter_cos_sim + weights

            interaction_cl_loss = self.cl_loss_fn(inter_cos_sim, inter_labels)
        else:
            q = batch["skills"]
            r = batch["responses"]

            attention_mask = batch["attention_mask"]

        q_embed = self.question_embed(q)
        i_embed = self.get_interaction_embed(q, r)

        x, y = q_embed, i_embed

        for block in self.question_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=True)

        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, apply_pos=True)

        for block in self.knoweldge_retriever:
            x, attn = block(mask=0, query=x, key=x, values=y, apply_pos=True)

        retrieved_knowledge = torch.cat([x, q_embed], dim=-1)

        output = torch.sigmoid(self.out(retrieved_knowledge)).squeeze()

        if self.training:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "cl_loss": 0.6 * question_cl_loss + 0.4 * interaction_cl_loss,
                "attn": attn,
            }
        else:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "attn": attn,
                "x": x,
            }

        return out_dict

    def alignment_and_uniformity(self, out_dict):
        return (
            out_dict["question_alignment"],
            out_dict["interaction_alignment"],
            out_dict["question_uniformity"],
            out_dict["interaction_uniformity"],
        )

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        cl_loss = torch.mean(out_dict["cl_loss"])
        mask = true > -1

        loss = self.loss_fn(pred[mask], true[mask]) + self.reg_cl * cl_loss

        return loss, len(pred[mask]), true[mask].sum().item()

    def get_interaction_embed(self, skills, responses):
        masked_responses = responses * (responses > -1).long()
        interactions = skills + self.num_skills * masked_responses
        return self.interaction_embed(interactions)


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], (
            "unrecognized pooling type %s" % self.pooler_type
        )

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def align_loss(x, y, alpha=2):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    x = F.normalize(x, dim=1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
