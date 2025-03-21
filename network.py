import torch
import torch.nn as nn
import math
import numpy as np
from torch import amp
from arguments import arg

# Computing attention scores: out
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()

        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim

        self.value_dim = embedding_dim
        self.key_dim = self.value_dim

        self.tanh_clipping = 10 # limit output range
        self.norm_factor = 1 / math.sqrt(self.key_dim) # predefined normalization factor, good idea

        # Learnable weight matrices for query and key projections
        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        """
        Initialize parameters using a uniform distribution in the range [-stdv, stdv].
        """
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                Compute single-head attention.
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim) -- maybe this is key tensor?
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return: att scores after log-softmax
                """
        if h is None:
            h = q # if no key provided, use query as key

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp (*important)

        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        # reshape tensors for matrix multiplication
        h_flat = h.reshape(-1, input_dim)  # ((batch_size*graph_size), input_dim)
        q_flat = q.reshape(-1, input_dim)  # ((batch_size*n_query), input_dim)

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        # Q * K^T
        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U = U-1e8*mask  # ??
            # U[mask.bool()] = -1e8
            U[mask.bool()] = -1e4
        attention = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        out = attention

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            U[mask.bool()] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc

        # Attention to value vectors
        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # reshape, project to output space
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim) # torch build-in layer norm

    def forward(self, input):
        return self.normalizer(input.reshape(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, mask=None):
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, mask=None):
        h0 = tgt # Save the original input for residual connection
        tgt = self.normalization1(tgt) # Apply first normalization
        memory = self.normalization1(memory) # Normalize encoder output
        h = self.multiHeadAttention(q=tgt, h=memory, mask=mask) # Apply multi-head attention
        h = h + h0 # Residual connection
        h1 = h # Save the intermediate output for the next residual connection
        h = self.normalization2(h) # Apply second normalization
        h = self.feedForward(h) # Apply feed-forward network
        h2 = h + h1 # Residual connection

        return h2


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)
        return tgt

# inject positional info into input embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) # prevent overfitting
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # store pe into model.state_dict(), but don't update

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.permute(1, 0, 2) # seq_len, batch_size, embedding_dim
        x = x + self.pe[:x.size(0)] # pe: max_len x b x embedding_dim
        return self.dropout(x).permute(1, 0, 2)


class AttentionNet(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_agents = arg.num_agents

        self.target_encoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.temporal_encoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.spatio_encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.spatio_decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)

        self.pointer = SingleHeadAttention(embedding_dim)

        self.spatio_pos_embedding = nn.Linear(32, embedding_dim)
        self.loc_embedding = nn.Linear(2, embedding_dim)
        self.belief_embedding = nn.Linear(4, embedding_dim)
        self.occupancy_embedding = nn.Linear(1, embedding_dim) # new
        self.timefusion_layer = nn.Linear(1, embedding_dim)
        self.distfusion_layer = nn.Linear(embedding_dim+1, embedding_dim)

        self.value_output = nn.Linear(embedding_dim, 1) # 每个agent独立


    def temporal_attention(self, history_inputs, temporal_mask, dt_inputs):
        """
        history_inputs: batch x #history x #nodes x inputdim
        temporal_mask: batch. Current step, indicate number of valid history, range(1,+inf)
        """
        feature_size = 4
        batch_size, history_size, graph_size, input_dim = history_inputs.shape
        target_size = (input_dim - 2) // feature_size

        history_inputs = history_inputs.reshape(-1, 1, input_dim)
        loc_feature = self.loc_embedding(history_inputs[:, :, :2])
        target_feature = []

        for i in range(target_size):
            feat = history_inputs[:, :, 2 + i * feature_size: 2 + (i + 1) * feature_size]
            feat_emb = self.belief_embedding(feat)
            target_feature.append(feat_emb)
        target_feature = torch.cat(target_feature, dim=1)
        embedded_feature = torch.cat((loc_feature, target_feature), dim=1)

        # Temporal encoding of history
        embedded_feature = self.target_encoder(embedded_feature[:, :1, :], embedded_feature)  # (bxgxh, 1, 128)
        embedded_feature = embedded_feature.reshape(batch_size, history_size, graph_size, self.embedding_dim)
        embedded_feature = embedded_feature.permute(0, 2, 1, 3).reshape(-1, history_size, self.embedding_dim) # (bx201)x10x128
        
        # Fuse time diff info
        dt_inputs = dt_inputs.unsqueeze(1).repeat(1, graph_size, 1, 1).reshape(-1, history_size, 1)  # (bx201)x10x1
        embedded_feature += self.timefusion_layer(dt_inputs)

        mask = torch.zeros((batch_size*graph_size, 1, history_size), dtype=torch.bool).to(history_inputs.device)
        if temporal_mask is not None:
            # Ensure temporal_mask does not exceed history_size
            temporal_mask = temporal_mask.clone()
            temporal_mask[temporal_mask > history_size] = history_size
            for ib in range(batch_size):
                valid_steps = temporal_mask[ib].item() if isinstance(temporal_mask[ib], torch.Tensor) else temporal_mask[ib]
                if valid_steps < history_size:
                    mask[ib * graph_size : (ib+1) * graph_size, 0, : (history_size - valid_steps)] = True

        embedded_temporal_feature = self.temporal_encoder(embedded_feature[:, -1:, :], embedded_feature, mask)
        embedded_temporal_feature = embedded_temporal_feature.reshape(batch_size, graph_size, self.embedding_dim)
        return embedded_temporal_feature


    def spatio_attention(self, embedded_feature, edge_inputs, dist_inputs, current_index, spatio_pos_encoding, spatio_mask=None):
        # 将 occupancy 信息加入节点特征
        batch_size, graph_size, _ = embedded_feature.size()
        occupancy = torch.zeros(batch_size, graph_size, device=embedded_feature.device)
        # 标记每个agent当前所在节点
        occupancy.scatter_(1, current_index.squeeze(-1).long(), 1)
        occupancy = occupancy.unsqueeze(-1)  # [batch_size, graph_size, 1]
        embedded_feature = embedded_feature + self.occupancy_embedding(occupancy)
        # 加入空间位置信息并进行图编码
        embedded_feature = embedded_feature + self.spatio_pos_embedding(spatio_pos_encoding)
        embedded_feature = self.spatio_encoder(embedded_feature)
        embedded_feature = self.distfusion_layer(torch.cat((embedded_feature, dist_inputs), dim=-1))

        # 获取每个agent当前节点的特征
        agent_indices = current_index.long()
        agent_idx_expanded = agent_indices.expand(-1, -1, self.embedding_dim)
        agent_current_features = embedded_feature.gather(dim=1, index=agent_idx_expanded)  # [batch_size, n_agents, embedding_dim]

        # 获取每个agent的邻居节点特征
        neighbor_size = edge_inputs.size(2)
        neighbor_idx = edge_inputs.gather(dim=1, index=agent_indices.expand(-1, -1, neighbor_size))
        embedded_feature_exp = embedded_feature.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        neighbor_idx_expanded = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim)
        all_agent_neighbors = torch.gather(embedded_feature_exp, 2, neighbor_idx_expanded)  # [batch_size, n_agents, neighbor_size, embedding_dim]

        # Pointer网络输入和mask构造
        h_flat = all_agent_neighbors.reshape(batch_size, -1, self.embedding_dim)  # [batch_size, n_agents*neighbor_size, embedding_dim]
        total_neighbors = self.n_agents * neighbor_size
        pointer_mask = torch.ones((batch_size, self.n_agents, total_neighbors), dtype=torch.bool, device=embedded_feature.device)
        for i in range(self.n_agents):
            pointer_mask[:, i, i * neighbor_size : (i+1) * neighbor_size] = False

        # Actor: pointer机制计算各agent邻居节点的选择概率 (logits)
        with amp.autocast(device_type='cuda'):
            logp_all = self.pointer(agent_current_features, h_flat, pointer_mask)
            # Critic: 独立价值估计，使用解码器计算每个agent当前状态的价值
            critic_query = agent_current_features.reshape(batch_size * self.n_agents, 1, self.embedding_dim)
            critic_memory = all_agent_neighbors.reshape(batch_size * self.n_agents, neighbor_size, self.embedding_dim)
            critic_out = self.spatio_decoder(critic_query, critic_memory)  # [batch*n_agents, 1, embedding_dim]
            critic_out = critic_out.squeeze(1)  # [batch*n_agents, embedding_dim]
            value_all_flat = self.value_output(critic_out)  # [batch*n_agents, 1]

        value_all = value_all_flat.view(batch_size, self.n_agents)  # [batch_size, n_agents]

        # 提取每个agent对应邻居的log概率
        idx = torch.arange(total_neighbors, device=logp_all.device, dtype=torch.long).view(self.n_agents, neighbor_size)
        logp_list = logp_all.gather(dim=2, index=idx.unsqueeze(0).expand(batch_size, -1, -1))
        return logp_list, value_all

    def forward(self, history_inputs, edge_inputs, dist_inputs, dt_inputs, current_index, spatio_pos_encoding, temporal_mask, spatio_mask=None):
        with amp.autocast(device_type='cuda'):
            embedded_temporal_feature = self.temporal_attention(history_inputs, temporal_mask, dt_inputs)
            logp_list, value = self.spatio_attention(embedded_temporal_feature, edge_inputs, dist_inputs, current_index, spatio_pos_encoding, spatio_mask)
        # logp_list: [batch_size, n_agents, neighbor_size]
        # value: [batch_size, n_agents]
        return logp_list, value


if __name__ == '__main__':
    pass
