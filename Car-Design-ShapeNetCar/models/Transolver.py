import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class DotProductAttention(nn.Module):
    def __init__(self, heads, dim_head, slice_num, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(torch.softmax(dots, dim=-1))
        return torch.matmul(attn, v)

class DotProductAttentionFlash(nn.Module):
    def __init__(self, heads, dim_head, slice_num, dropout=0.):
        super().__init__()
        self.dropout_p = dropout

    def forward(self, q, k, v):
        dropout_p = getattr(self, 'dropout_p', 0.) if self.training else 0.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

class MahalanobisAttention(nn.Module):
    def __init__(self, heads, dim_head, slice_num, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.L = nn.Parameter(torch.eye(dim_head).unsqueeze(0).repeat(heads, 1, 1)) # H D D

    def forward(self, q, k, v):
        qL = torch.einsum('bhgd,hde->bhge', q, self.L)
        kL = torch.einsum('bhgd,hde->bhge', k, self.L)
        dots = torch.matmul(qL, kL.transpose(-1, -2)) * self.scale
        attn = self.dropout(torch.softmax(dots, dim=-1))
        return torch.matmul(attn, v)

class GaussianAttention(nn.Module):
    def __init__(self, heads, dim_head, slice_num, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.log_sigma = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(self, q, k, v):
        q_sq = (q * q).sum(-1, keepdim=True) # B H G 1
        k_sq = (k * k).sum(-1).unsqueeze(-2) # B H 1 G
        dist_sq = q_sq + k_sq - 2 * torch.matmul(q, k.transpose(-1, -2))
        sigma_sq = torch.exp(2 * self.log_sigma)
        scores = -dist_sq / (2 * sigma_sq + 1e-6)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(attn, v)

class BilinearAttention(nn.Module):
    def __init__(self, heads, dim_head, slice_num, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.eye(dim_head).unsqueeze(0).repeat(heads, 1, 1)) # H D D

    def forward(self, q, k, v):
        Wk = torch.einsum('hde,bhge->bhgd', self.W, k)    # W k
        dots = torch.matmul(q, Wk.transpose(-1, -2)) * self.scale
        attn = self.dropout(torch.softmax(dots, dim=-1))
        return torch.matmul(attn, v)

class LinearAttention(nn.Module):
    def __init__(self, heads, dim_head, slice_num, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def feature_map(x):
        return F.elu(x) + 1

    def forward(self, q, k, v):
        qf = self.feature_map(q) # B H G D
        kf = self.feature_map(k) # B H G D
        kv = torch.einsum('bhgd,bhge->bhde', kf, v) # B H D D
        z = 1.0 / (torch.einsum('bhgd,bhd->bhg', qf, kf.sum(dim=2)) + 1e-6)
        out = torch.einsum('bhgd,bhde->bhge', qf, kv) * z.unsqueeze(-1)
        return self.dropout(out)


class MixtureAttention(nn.Module):
    def __init__(self, heads, dim_head, slice_num, dropout=0., n_mix=4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.n_mix = n_mix
        self.Wk = nn.Parameter(
            torch.eye(dim_head).reshape(1, 1, dim_head, dim_head).repeat(heads, n_mix, 1, 1)
        )  # H K D D
        self.gate = nn.Linear(dim_head, n_mix)

    def forward(self, q, k, v):
        Wk = torch.einsum('hkde,bhge->bhkgd', self.Wk, k) # B H K G D
        dots = torch.einsum('bhid,bhkjd->bhkij', q, Wk) * self.scale
        comp = torch.softmax(dots, dim=-1) # B H K G G
        pi = torch.softmax(self.gate(q), dim=-1) # B H G K
        pi = pi.permute(0, 1, 3, 2).unsqueeze(-1) # B H K G 1
        attn = (pi * comp).sum(dim=2) # B H G G
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

ATTENTION_FUNCTIONS = {
    'dot_product': DotProductAttention,
    'dot_product_flash': DotProductAttentionFlash,
    'mahalanobis': MahalanobisAttention,
    'gaussian': GaussianAttention,
    'bilinear': BilinearAttention,
    'linear': LinearAttention,
    'mixture': MixtureAttention
}
def build_attention(attn_type, heads, dim_head, slice_num, dropout=0.):
    if attn_type not in ATTENTION_FUNCTIONS:
        raise ValueError(
            f"Unknown attn_type '{attn_type}'. Choose from {list(ATTENTION_FUNCTIONS)}"
        )
    return ATTENTION_FUNCTIONS[attn_type](heads, dim_head, slice_num, dropout=dropout)

class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, attn_type='dot_product'):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.attn_type = attn_type

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.attn = build_attention(attn_type, heads, dim_head, slice_num, dropout=dropout)
        
        self.slice_num = slice_num
        self.cluster_centers = nn.Parameter(torch.empty(self.heads, self.slice_num, self.dim_head))
        nn.init.orthogonal_(self.cluster_centers)
        with torch.no_grad():
            self.cluster_centers.data.mul_(math.sqrt(self.dim_head) * 0.5)
        self.log_precision_diag = nn.Parameter(torch.zeros(self.heads, self.slice_num, self.dim_head))

        self.slice_norm1 = nn.LayerNorm(self.dim_head)
        self.slice_norm2 = nn.LayerNorm(self.dim_head)
        self.slice_mlp = nn.Sequential(nn.Linear(self.dim_head, self.dim_head * 4), nn.GELU(), nn.Linear(self.dim_head * 4, self.dim_head))
    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C

        precision = torch.nn.functional.softplus(self.log_precision_diag)
        centers = self.cluster_centers
        x_sq = x_mid.pow(2)
        term_quad = torch.matmul(x_sq, precision.transpose(-1, -2))
        weighted_centers = centers * precision
        term_lin = torch.matmul(x_mid, weighted_centers.transpose(-1, -2))
        term_bias = (centers.pow(2) * precision).sum(dim=-1).view(1, self.heads, 1, self.slice_num)
        sq_dist = term_quad - 2 * term_lin + term_bias
        log_det = torch.sum(torch.log(precision + 1e-6), dim=-1).view(1, self.heads, 1, self.slice_num)
        slice_logits = -0.5 * sq_dist + 0.5 * log_det
        slice_weights = self.softmax(slice_logits / torch.clamp(self.temperature, min=0.1, max=5))

        slice_norm = slice_weights.sum(2) # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)

        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)

        res = slice_token
        slice_token = self.slice_norm1(slice_token)

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        slice_token = res + self.attn(q_slice_token, k_slice_token, v_slice_token) # B H G D
        res = slice_token
        out_slice_token = res + self.slice_mlp(self.slice_norm2(slice_token))

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()
        act_func = ACTIVATION.get(act, nn.GELU)
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_func())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act_func()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = (self.linears[i](x) + x) if self.res else self.linears[i](x)
        return self.linear_post(x)


class Transolver_block(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, dropout: float, act='gelu', mlp_ratio=4, last_layer=False,
        out_dim=1, slice_num=32, attn_type='dot_product'):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num, attn_type=attn_type)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class Model(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0, n_head=8, act='gelu', mlp_ratio=1, fun_dim=1,
         out_dim=4, slice_num=32, ref=8, unified_pos=False, attn_type='dot_product'):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D_TripleHead'
        self.ref = ref
        self.unified_pos = unified_pos
        self.attn_type = attn_type

        # ENTRANCE: Add +1 to fun_dim to account for the binary "is_surface" indicator
        if self.unified_pos:
            self.preprocess = MLP((fun_dim + 1) + self.ref ** 3, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP((fun_dim + 1) + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act,
                            mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num, attn_type=attn_type,
                            last_layer=(_ == n_layers - 1)) for _ in range(n_layers)])

        self.ln_final = nn.LayerNorm(n_hidden)

        self.head_vol_u = nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, 1)
        )

        self.head_vol_v = nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, 1)
        )

        self.head_vol_w = nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, 1)
        )

        self.head_surf_u = nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, 1)
        )

        self.head_surf_v = nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, 1)
        )

        self.head_surf_w = nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, 1)
        )

        # 3. Surface Pressure (1 channel)
        self.head_surf_press = nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, 1)
        )

        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        batchsize = my_pos.shape[0]
        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)
        pos = torch.sqrt(torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)).reshape(batchsize,
                                                      my_pos.shape[
                                                        1],
                                                      self.ref ** 3).contiguous()
        return pos

    def forward(self, data):
        cfd_data, geom_data = data
        x, fx, T = cfd_data.x, None, None

        # Prepare Batch shape [1, N, C]
        surf_mask = cfd_data.surf

        # Create a numeric indicator feature: 1.0 for surface, 0.0 for volume
        is_surf_feature = surf_mask.float().unsqueeze(-1)

        # Append the indicator to the node features
        x = torch.cat((x, is_surf_feature), dim=-1)
        x = x[None, :, :]
        if next(self.parameters()).dtype == torch.float16:
            x=x.half()

        if self.unified_pos:
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        fx = self.ln_final(fx) # [1, N, d]

        out = None
        if next(self.parameters()).dtype == torch.float16:
            out = torch.zeros(1, fx.shape[1], 4, device=fx.device, dtype=torch.float16)
        else:
            out = torch.zeros(1, fx.shape[1], 4, device=fx.device)

        out[0, ~surf_mask, 0:1] = self.head_vol_u(fx[0, ~surf_mask, :])
        out[0, ~surf_mask, 1:2] = self.head_vol_v(fx[0, ~surf_mask, :])
        out[0, ~surf_mask, 2:3] = self.head_vol_w(fx[0, ~surf_mask, :])

        out[0, surf_mask, 0:1] = self.head_surf_u(fx[0, surf_mask, :])
        out[0, surf_mask, 1:2] = self.head_surf_v(fx[0, surf_mask, :])
        out[0, surf_mask, 2:3] = self.head_surf_w(fx[0, surf_mask, :])

        out[0, surf_mask, 3:4] = self.head_surf_press(fx[0, surf_mask, :])

        return out[0]
