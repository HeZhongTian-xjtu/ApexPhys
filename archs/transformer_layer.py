from einops import rearrange
from torch import nn
from torch.nn import functional as F


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention with Conv3d"""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()

        self.proj_q = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )

        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads

    def forward(self, x):  # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        [B, P, C] = x.shape
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)  # [B, dim, 40, 4, 4]

        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim//2]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim//2]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim//2]

        # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')
        q, k, v = [
            rearrange(x, "b n (h d) -> b h n d", h=self.n_heads) for x in [q, k, v]
        ]
        # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N) -softmax-> (B, H, N, N)
        # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')
        attn = q @ k.transpose(-2, -1)
        # print(f'attn: {attn.shape}')

        attn = self.drop(F.softmax(attn, dim=-1))  # [B, H, S, S]
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        out = attn @ v
        # -merge-> (B, S, D)
        # print(f'out: {out.shape}')
        out = rearrange(out, "b h n d -> b n (h d)")
        # print(f'out: {out.shape}')
        return out


class FeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )

        # self.STConv = nn.Sequential(
        #     nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
        #     nn.BatchNorm3d(ff_dim),
        #     nn.ELU(),
        # )

        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):  # [B, 4*4*40, 128]
        [B, P, C] = x.shape
        # x = x.transpose(1, 2).view(B, C, 40, 4, 4)      # [B, dim, 40, 4, 4]
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)  # [B, dim, 40, 4, 4]
        x = self.fc1(x)  # x [B, ff_dim, 40, 4, 4]
        # x = self.STConv(x)		          # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)  # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]

        return x


class ViTBlock(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = FeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        Atten = self.attn(self.norm1(x))
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class ViTBackbone(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ViTBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
