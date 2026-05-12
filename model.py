import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler, WeightedRandomSampler, TensorDataset

def _get_act(name: str) -> nn.Module:
    return {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}[name.lower()]()

def apply_modality_dropout(stack, p_drop=0.15, training=True):
    """ Randomly zero out entire modality token(s) during training; returns stack with some modality tokens zeroed, shape unchanged (B, S, E) """
    if not training or p_drop == 0.0:
        return stack
    B, S, E = stack.shape
    # sample a drop mask: shape (B, S, 1), broadcast over E
    mask = (torch.rand(B, S, 1, device=stack.device) > p_drop).float()
    # ensure at least one modality is kept per sample
    keep_count = mask.sum(dim=1, keepdim=True)  # (B, 1, 1)
    # where all modalities would be dropped, fall back to keeping all
    safe_mask = torch.where(keep_count == 0, torch.ones_like(mask), mask)
    return stack * safe_mask

# ENCODERS
class FFNEncoder(nn.Module):
    """ Simple feed-forward encoder used for each modality """
    def __init__(self, input_dim, mparams, out_norm=True):
        super().__init__()
        layers = []
        act_cls = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}[mparams['act'].lower()]
        if mparams['input_drop'] != 0:
            layers.append(nn.Dropout(mparams['input_drop']))
        prev = input_dim
        for h, p in zip(mparams['n'], mparams['dropout']):
            layers.append(nn.Linear(prev, h))
            if mparams['norm']:
                layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            if p != 0:
                layers.append(nn.Dropout(p))
            prev = h
        # final projection without non-linearity (leave embedding space linear)
        if prev != mparams['out']:
            layers.append(nn.Linear(prev, mparams['out']))
        if out_norm:
            layers.append(nn.LayerNorm(mparams['out']))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (batch, input_dim) → (batch, out)

# FUSION MODULES
class ModalityEmbedding(nn.Module):
    """ Adds a learned type embedding to each modality token """
    def __init__(self, n_modalities: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_modalities, embed_dim)

    def forward(self, stack): # stack: (B, S, E)
        idx = torch.arange(stack.size(1), device=stack.device)
        return stack + self.emb(idx).unsqueeze(0) # (B, S, E) + (1, S, E)

class GatedModalityFusion(nn.Module):
    """ Attention fusion with per-modality confidence gates and learned type embeddings
      1. Add learned modality-type embeddings (modality identity)
      2. Compute per-modality sigmoid gate from each token's embedding; suppresses noisy modalities, amplifies informative ones
      3. Apply modality dropout during training
      4. Multi-head self-attention over gated token sequence
      5. Gate-weighted average pooling
      6. Small post-attention adapter with GELU """
    def __init__(self, fparams, aparams, n_modalities=3, mod_dropout=0.15):
        super().__init__()
        embed_dim = int(fparams['embed'])
        self.mod_emb = ModalityEmbedding(n_modalities, embed_dim)
        # Per-modality scalar gate: (B, S, E) → (B, S, 1)
        self.gate = nn.Linear(embed_dim, 1)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=aparams['n_heads'], dropout=aparams['dropout'], batch_first=True)
        self.post = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)) # GELU instead of ReLU to avoid dead neurons
        self.mod_dropout_p = mod_dropout

    def forward(self, mod_embeddings: torch.Tensor):
        # mod_embeddings: (B, S, E), S = number of active modalities
        x = self.mod_emb(mod_embeddings)    # add type embeddings
        gates = torch.sigmoid(self.gate(x)) # per-modality scalar confidence (B, S, 1)
        x_gated = x * gates # each token soft-scaled by its gate
        x_gated = apply_modality_dropout(x_gated, self.mod_dropout_p, self.training)
        gate_w = gates.squeeze(-1) # (B, S)
        gate_w = gate_w / (gate_w.sum(dim=1, keepdim=True) + 1e-8) # normalize
        attn_out, attn_weights = self.mha(x_gated, x_gated, x_gated, need_weights=True, average_attn_weights=False)
        fused = (attn_out * gate_w.unsqueeze(-1)).sum(dim=1) # (B, E)
        fused = self.post(fused)
        return fused, attn_weights

class ConcatFusion(nn.Module):
    """ No-attention baseline: concatenate modality embeddings, project back to embed_dim via MLP
      1. Apply modality dropout during training (randomly zero out modalities)
      2. Flatten (B, S, E) → (B, S*E)
      3. MLP: [S*E → embed_dim → embed_dim] with GELU, dropout, LayerNorm """
    def __init__(self, fparams, aparams, n_modalities=3, mod_dropout=0.15):
        super().__init__()
        embed_dim = int(fparams['embed'])
        in_dim = n_modalities * embed_dim
        self.mod_dropout_p = mod_dropout
        self.mlp = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.GELU(), nn.Dropout(aparams['dropout']), nn.LayerNorm(embed_dim))
 
    def forward(self, mod_embeddings):
        # mod_embeddings: (B, S, E)
        x = apply_modality_dropout(mod_embeddings, self.mod_dropout_p, self.training)
        x = x.flatten(start_dim=1) # (B, S*E)
        fused = self.mlp(x) # (B, embed_dim)
        return fused, None

# PREDICTION HEAD
class PredictionHead(nn.Module):
    def __init__(self, fparams):
        super().__init__()
        act_cls = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}[fparams['act'].lower()]
        layers = []
        prev = fparams['embed']
        for h, p in zip(fparams['n'], fparams['dropout']):
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if p != 0:
                layers.append(nn.Dropout(p))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# PLASMA
class PLASMA(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.version = params.version
        self.params = params
        # modality encoders
        outnorm = True
        if 'C' in self.version:
            self.enc_clin = FFNEncoder(len(params.features[0]), params.clin, out_norm=outnorm)
        if 'E' in self.version:
            self.enc_expr = FFNEncoder(len(params.features[1]), params.expr, out_norm=outnorm)
        if 'M' in self.version:
            self.enc_mut = FFNEncoder(len(params.features[2]), params.mut, out_norm=outnorm)
        # post-encoder modules
        if len(self.version) > 1:
            _enc_params = {}
            if 'C' in self.version: _enc_params['C'] = params.clin
            if 'E' in self.version: _enc_params['E'] = params.expr
            if 'M' in self.version: _enc_params['M'] = params.mut
            # gated fusion with learned modality embeddings (encoder out dims must equal params.fus['embed'])
            n_mod = len(self.version)
            if params.att['use']:
                self.fusion = GatedModalityFusion(params.fus, params.att, n_modalities=n_mod, mod_dropout=params.mod_dropout)
            else:
                self.fusion = ConcatFusion(params.fus, params.att, n_modalities=n_mod, mod_dropout=params.mod_dropout)
            self.head = PredictionHead(params.fus)
            # auxiliary per-modality classification heads (ended up setting this weight to 0, so this is inconsequential)
            self.aux_heads = nn.ModuleDict()
            for key, ep in _enc_params.items():
                self.aux_heads[key] = nn.Linear(ep['out'], 1)
        else:
            self.head = PredictionHead(params.fus)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                # small init for modality embeddings to not overwhelm encoder signal at start
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x_c=None, x_e=None, x_m=None):
        """ Returns logits (raw binary logits), probs (sigmoid probs), attn weights (B, n_heads, S, S), aux_logits (dict, not used) """
        enc_c = enc_e = enc_m = None
        if 'C' in self.version: enc_c = self.enc_clin(x_c)
        if 'E' in self.version: enc_e = self.enc_expr(x_e)
        if 'M' in self.version: enc_m = self.enc_mut(x_m)
        aux_logits = {}
        if len(self.version) == 1:
            # single-modality: encoder → head, no fusion (head matches the multi-modality variant so the post-encoder pipeline is identical across all comparisons)
            attn_weights = None
            if self.version == 'C':   logits = self.head(enc_c)
            elif self.version == 'E': logits = self.head(enc_e)
            elif self.version == 'M': logits = self.head(enc_m)
        else:
            # compute auxiliary logits per modality (before projection)
            if 'C' in self.version:
                aux_logits['C'] = self.aux_heads['C'](enc_c).squeeze(-1)
            if 'E' in self.version:
                aux_logits['E'] = self.aux_heads['E'](enc_e).squeeze(-1)
            if 'M' in self.version:
                aux_logits['M'] = self.aux_heads['M'](enc_m).squeeze(-1)
            to_stack = []
            if 'C' in self.version: to_stack.append(enc_c)
            if 'E' in self.version: to_stack.append(enc_e)
            if 'M' in self.version: to_stack.append(enc_m)
            stack = torch.stack(to_stack, dim=1) # (B, N_mod, embed_dim)
            fused, attn_weights = self.fusion(stack)
            logits = self.head(fused)
        logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs, attn_weights, aux_logits