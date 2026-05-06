import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ViEEG.ViEEG_utils import *
from einops.layers.torch import Rearrange
from torch import Tensor


class ViEEG(nn.Module):
    def __init__(
            self,
            time_length=250,
            n_channel=63,
            hidden_dim=250,
            n_layer=1,
            n_head=1,
            filter_time_length=5,
            n_filters=1,
            dropout_rate=0.5,
            device='cuda:0'
    ):
        super(ViEEG, self).__init__()
        self.T = time_length
        self.C = n_channel
        self.n_layer = n_layer
        self.n_head = n_head

        self.STCConv1 = STC(n_channel, dropout_rate)
        self.STCConv2 = STC(n_channel, dropout_rate)
        self.STCConv3 = STC(n_channel, dropout_rate)

        proj_dim = 1024
        if hidden_dim == 512:
            f_size = 3520
        elif hidden_dim == 256:
            f_size = 1480
        elif hidden_dim == 250:
            f_size = 1440

        if self.n_layer > 0:
            self.Transor1 = CrossEncoder(36, 36*4, n_layer, n_head, dropout_rate)
            self.Transor2 = CrossEncoder(36, 36*4, n_layer, n_head, dropout_rate)

        self.project1 = nn.Sequential(nn.Linear(f_size, proj_dim),
                                    ResidualAdd(nn.Sequential(
                                        nn.GELU(),
                                        nn.Linear(proj_dim, proj_dim),
                                        nn.Dropout(dropout_rate),
                                    )),
                                    nn.LayerNorm(proj_dim),
                                    )
        self.project2 = nn.Sequential(nn.Linear(f_size, proj_dim),
                                    ResidualAdd(nn.Sequential(
                                        nn.GELU(),
                                        nn.Linear(proj_dim, proj_dim),
                                        nn.Dropout(dropout_rate),
                                    )),
                                    nn.LayerNorm(proj_dim),
                                    )
        self.project3 = nn.Sequential(nn.Linear(f_size, proj_dim),
                                    ResidualAdd(nn.Sequential(
                                        nn.GELU(),
                                        nn.Linear(proj_dim, proj_dim),
                                        nn.Dropout(dropout_rate),
                                    )),
                                    nn.LayerNorm(proj_dim),
                                    )

    def forward(self, x):
        v_x = self.STCConv1(x)
        m_x = self.STCConv2(x)
        b_x = self.STCConv3(x)
        v_x = torch.transpose(v_x, 1, 2)
        m_x = torch.transpose(m_x, 1, 2)
        b_x = torch.transpose(b_x, 1, 2)
        if self.n_layer > 0:
            b_x = b_x
            m_x = self.Transor1(m_x, b_x, m_x)
            v_x = self.Transor2(v_x, m_x, v_x)

        v_x = v_x.contiguous().view(v_x.size(0), -1)
        m_x = m_x.contiguous().view(m_x.size(0), -1)
        b_x = b_x.contiguous().view(b_x.size(0), -1)

        v_x = self.project1(v_x)
        m_x = self.project2(m_x)
        b_x = self.project3(b_x)

        return v_x, m_x, b_x


class STC(nn.Module):
    def __init__(self, n_channel=63, dropout_rate=0.5):
        super(STC, self).__init__()
        con_size = 40
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, con_size, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(con_size),
            nn.ELU(),
            nn.Conv2d(con_size, con_size, (n_channel, 1), (1, 1)),
            nn.BatchNorm2d(con_size),
            nn.ELU(),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(con_size, con_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, input_dim=768):
        super(AdaptiveFeatureFusion, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim * input_dim, 2)  # 用于从矩阵生成标量权重

    def forward(self, v_x, s_x):
        batch_size, a = v_x.size()

        # Step 1: 计算交互矩阵 M
        M = torch.matmul(v_x.unsqueeze(2), s_x.unsqueeze(1))  # (batch_size, a, a)

        # Step 2: 将矩阵 M 展平成向量并通过全连接层生成权重
        M_flat = M.view(batch_size, -1)  # (batch_size, a * a)
        weights = F.softmax(self.fc(M_flat), dim=-1)  # (batch_size, 2)

        # Step 3: 提取标量权重 a, b
        a, b = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)  # (batch_size, 1)

        # Step 4: 融合特征
        fused_x = a * v_x + b * s_x  # (batch_size, a)

        return fused_x, (a, b)


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=512, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


if __name__ == "__main__":
    model = ViEEG()
    input_data = torch.randn(1, 1, 63, 250)
    v_eeg_features, m_eeg_features, b_eeg_features = model(input_data)
