# network.py
import torch
import torch.nn as nn
import math
import numpy as np
from torch import amp # For automatic mixed precision
from arguments import arg

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.key_dim = embedding_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q
        batch_size, h_len, _ = h.size()
        n_query = q.size(1)

        h_flat = h.reshape(-1, self.input_dim)
        q_flat = q.reshape(-1, self.input_dim)

        shape_k = (batch_size, h_len, self.key_dim)
        shape_q = (batch_size, n_query, self.key_dim)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(h_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U[mask.bool()] = -float('inf')

        attention_log_probs = torch.log_softmax(U, dim=-1)
        return attention_log_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        assert embedding_dim % n_heads == 0, "Embedding dimension must be divisible by n_heads"
        self.head_dim = embedding_dim // n_heads

        self.fc_q = nn.Linear(embedding_dim, embedding_dim)
        self.fc_k = nn.Linear(embedding_dim, embedding_dim)
        self.fc_v = nn.Linear(embedding_dim, embedding_dim)
        self.fc_o = nn.Linear(embedding_dim, embedding_dim)
        self.norm_factor = 1 / math.sqrt(self.head_dim)

    def forward(self, q, h=None, mask=None):
        if h is None:
            k, v = q, q # Self-attention
        else:
            k, v = h, h # Cross-attention where h provides keys and values

        batch_size = q.shape[0]
        query_len = q.shape[1]
        key_len = k.shape[1]

        Q = self.fc_q(q).view(batch_size, query_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_k(k).view(batch_size, key_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_v(v).view(batch_size, key_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_factor # [B, H, Q_len, K_len]

        if mask is not None:
            # Mask shape should be [B, Q_len, K_len] or broadcastable
            mask_expanded = mask.unsqueeze(1).expand_as(energy) # [B, 1, Q_len, K_len]
            energy[mask_expanded.bool()] = -float('inf')
        
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V) # [B, H, Q_len, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous() # [B, Q_len, H, head_dim]
        x = x.view(batch_size, query_len, self.embedding_dim) # [B, Q_len, EmbDim]
        x = self.fc_o(x)
        return x

class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input_tensor):
        return self.normalizer(input_tensor) # LayerNorm can handle various shapes directly


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, n_head)
        self.norm1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.norm2 = Normalization(embedding_dim)

    def forward(self, src, src_mask=None):
        att_out = self.self_attention(q=src, mask=src_mask)
        src = self.norm1(src + att_out) # Add then Norm
        
        ff_out = self.feedForward(src)
        src = self.norm2(src + ff_out) # Add then Norm
        return src


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, n_head)
        self.norm1 = Normalization(embedding_dim)
        self.cross_attention = MultiHeadAttention(embedding_dim, n_head)
        self.norm2 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.norm3 = Normalization(embedding_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-Attention on target sequence
        self_att_out = self.self_attention(q=tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self_att_out)

        # Cross-Attention (Encoder-Decoder Attention)
        cross_att_out = self.cross_attention(q=tgt, h=memory, mask=memory_mask)
        tgt = self.norm2(tgt + cross_att_out)

        # Feed-Forward Network
        ff_out = self.feedForward(tgt)
        tgt = self.norm3(tgt + ff_out)
        return tgt


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, n_head) for _ in range(n_layer)])
        self.norm = Normalization(embedding_dim)

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for _ in range(n_layer)])
        self.norm = Normalization(embedding_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(tgt)


class PositionalEncoding(nn.Module): # No changes needed here
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionNet(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_agents = arg.num_agents

        self.temporal_encoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.spatio_encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)

        self.pointer = SingleHeadAttention(embedding_dim) # Actor's action selection head

        self.spatio_pos_embedding = nn.Linear(32, embedding_dim)
        self.loc_embedding = nn.Linear(2, embedding_dim)
        self.belief_embedding = nn.Linear(4, embedding_dim)
        self.occupancy_embedding = nn.Linear(1, embedding_dim)
        self.timefusion_layer = nn.Linear(1, embedding_dim)
        self.distfusion_layer = nn.Linear(embedding_dim + 1, embedding_dim)

        max_num_targets = arg.target_size[1] if isinstance(arg.target_size, tuple) else arg.target_size
        self.input_projection = nn.Linear((max_num_targets + 1) * embedding_dim, embedding_dim)
        
        # Decentralized Critic Head (shared parameters)
        # Takes individual agent's processed feature as input
        self.agent_value_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 128), # Hidden layer
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single value for that agent
        )

    def temporal_attention(self, history_inputs_pooled, temporal_mask_pooled, dt_inputs_pooled):
        # history_inputs_pooled: [B, PooledHistLen, NumNodes, NodeFeatureDim]
        # temporal_mask_pooled: [B] or [B,1] - valid length of POOLED history
        # dt_inputs_pooled: [B, PooledHistLen, 1]

        batch_size, pooled_history_len, num_nodes, node_feature_dim = history_inputs_pooled.shape
        num_actual_targets = (node_feature_dim - 2 - 1) // 4 # 2 for loc, 1 for occupancy

        history_reshaped = history_inputs_pooled.permute(0, 2, 1, 3).contiguous().view(
            batch_size * num_nodes, pooled_history_len, node_feature_dim
        )

        loc_feat = self.loc_embedding(history_reshaped[:, :, :2])
        occ_feat_this_node = self.occupancy_embedding(history_reshaped[:, :, -1:])
        fused_loc_and_own_occ = loc_feat + occ_feat_this_node

        target_belief_embeddings_list = []
        for i in range(num_actual_targets):
            start_idx = 2 + i * 4
            end_idx = 2 + (i + 1) * 4
            belief_feat_i = history_reshaped[:, :, start_idx:end_idx]
            target_belief_embeddings_list.append(self.belief_embedding(belief_feat_i))
        
        max_targets_config = arg.target_size[1] if isinstance(arg.target_size, tuple) else arg.target_size
        combined_target_feat_padded = torch.zeros(
            history_reshaped.size(0), pooled_history_len, max_targets_config * self.embedding_dim,
            device=history_inputs_pooled.device
        )
        if target_belief_embeddings_list:
            current_targets_emb = torch.cat(target_belief_embeddings_list, dim=2)
            combined_target_feat_padded[:, :, :current_targets_emb.shape[2]] = current_targets_emb
            
        combined_node_history_features = torch.cat((fused_loc_and_own_occ, combined_target_feat_padded), dim=2)
        projected_history_features = self.input_projection(combined_node_history_features)

        dt_expanded_for_nodes = dt_inputs_pooled.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        dt_reshaped = dt_expanded_for_nodes.contiguous().view(batch_size * num_nodes, pooled_history_len, 1)
        projected_history_features = projected_history_features + self.timefusion_layer(dt_reshaped)

        # Create mask for temporal_encoder (cross-attention memory mask)
        # query is current step, memory is all pooled history steps
        cross_attention_mask_temporal = torch.zeros(
            batch_size * num_nodes, 1, pooled_history_len, dtype=torch.bool, device=history_inputs_pooled.device
        )
        if temporal_mask_pooled is not None:
            for ib in range(batch_size):
                valid_pooled_len = temporal_mask_pooled[ib].item() # This is already the pooled length
                if valid_pooled_len < pooled_history_len:
                    cross_attention_mask_temporal[ib*num_nodes:(ib+1)*num_nodes, 0, valid_pooled_len:] = True
        
        current_features_query = projected_history_features[:, -1:, :] # Query from the latest pooled history step
        
        temporal_encoded_output = self.temporal_encoder(
            tgt=current_features_query, 
            memory=projected_history_features, 
            tgt_mask=None, 
            memory_mask=cross_attention_mask_temporal
        )
        temporal_encoded_output = temporal_encoded_output.squeeze(1)
        return temporal_encoded_output.view(batch_size, num_nodes, self.embedding_dim)

    def spatio_attention(self, embedded_temporal_feature, edge_inputs, dist_inputs, current_index, spatio_pos_encoding, spatio_mask_from_worker=None):
        batch_size, num_nodes, _ = embedded_temporal_feature.shape

        occupancy_now = torch.zeros(batch_size, num_nodes, 1, device=embedded_temporal_feature.device)
        for b in range(batch_size):
            occupancy_now[b, current_index[b,:,0].long(), 0] = 1.0 
            
        feature_for_spatio_encoder = embedded_temporal_feature + \
                                     self.occupancy_embedding(occupancy_now) + \
                                     self.spatio_pos_embedding(spatio_pos_encoding)
        
        # spatio_mask for encoder: e.g. from A_matrix, if nodes cannot attend to all other nodes.
        # For now, assume full self-attention or use spatio_mask_from_worker if it's for node-to-node attention.
        # The original spatio_mask was [B, graph_size+1, K_size] for pointer.
        # If spatio_encoder needs a mask of shape [B, NumNodes, NumNodes], it needs to be constructed.
        # Let's assume no mask for spatio_encoder for now, as in original.
        spatio_encoded_features = self.spatio_encoder(feature_for_spatio_encoder, src_mask=None)

        fused_spatio_features = self.distfusion_layer(
            torch.cat((spatio_encoded_features, dist_inputs), dim=-1)
        )

        agent_indices_long = current_index.long()
        agent_idx_expanded_for_gather = agent_indices_long.expand(-1, -1, self.embedding_dim)
        agent_queries = fused_spatio_features.gather(dim=1, index=agent_idx_expanded_for_gather)

        # --- Actor (Pointer Network) Part ---
        current_agent_nodes_edge_inputs = edge_inputs.gather(
            dim=1, 
            index=agent_indices_long.expand(-1, -1, arg.k_size)
        )
        expanded_fused_spatio_for_neighbor_gather = fused_spatio_features.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        expanded_neighbor_indices_for_gather = current_agent_nodes_edge_inputs.unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim)
        
        all_agents_neighbor_features = torch.gather(
            expanded_fused_spatio_for_neighbor_gather, 
            dim=2,
            index=expanded_neighbor_indices_for_gather
        )
        h_for_pointer = all_agents_neighbor_features.view(batch_size, self.n_agents * arg.k_size, self.embedding_dim)
        
        pointer_keys_mask = torch.ones(batch_size, self.n_agents, self.n_agents * arg.k_size, dtype=torch.bool, device=agent_queries.device)
        for i in range(self.n_agents):
            start_key_idx = i * arg.k_size
            end_key_idx = (i + 1) * arg.k_size
            pointer_keys_mask[:, i, start_key_idx:end_key_idx] = False
        
        # logp_all_keys will be [B, NumAgents, NumAgents*K_size] (log_softmax output)
        logp_all_keys = self.pointer(q=agent_queries, h=h_for_pointer, mask=pointer_keys_mask)
        
        final_logp_list = []
        for i in range(self.n_agents):
            start_key_idx = i * arg.k_size
            end_key_idx = (i + 1) * arg.k_size
            final_logp_list.append(logp_all_keys[:, i, start_key_idx:end_key_idx]) 
        logp_list_for_agents = torch.stack(final_logp_list, dim=1) # [B, NumAgents, K_size]

        # --- Decentralized Critic Part ---
        # agent_queries is [B, NumAgents, EmbDim]
        agent_queries_reshaped = agent_queries.contiguous().view(-1, self.embedding_dim) # [B * NumAgents, EmbDim]
        individual_agent_values = self.agent_value_head(agent_queries_reshaped) # [B * NumAgents, 1]
        individual_agent_values = individual_agent_values.view(batch_size, self.n_agents) # [B, NumAgents]
        
        return logp_list_for_agents, individual_agent_values

    def forward(self, history_inputs_pooled, edge_inputs, dist_inputs, dt_inputs_pooled, 
                current_index, spatio_pos_encoding, temporal_mask_pooled, spatio_mask_worker=None):
        # Inputs are assumed to be pooled from worker/driver
        with amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            embedded_temporal_feature_per_node = self.temporal_attention(
                history_inputs_pooled, 
                temporal_mask_pooled, # Mask for pooled history length
                dt_inputs_pooled
            )
            
            logp_list, individual_values = self.spatio_attention(
                embedded_temporal_feature_per_node, 
                edge_inputs, 
                dist_inputs, 
                current_index, 
                spatio_pos_encoding,
                spatio_mask_worker # This is the [B, G+1, K] mask from worker, used by pointer
            )
        return logp_list, individual_values