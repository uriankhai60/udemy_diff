import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1) # emb: 0.2971077539347156
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb) # 1.0 ~ 7.4e-05
#   print(emb.device)
  emb = timesteps.type(torch.float32)[:, None] * emb[None, :]
  emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = torch.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == (timesteps.shape[0], embedding_dim), f"{emb.shape} == [{timesteps.shape[0]}, {embedding_dim}]"
  return emb


class Downsample(nn.Module):
    def __init__(self, C:int):
        '''
        :param C (int): num of input and output channel
        '''
        super().__init__()
        self.conv = nn.Conv2d(C, C, 3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        assert x.shape == (B, C, H//2, W//2), f"x.shape == ({B}, {C}, {H//2}, {W//2})"
        return x
    

class Upsample(nn.Module):
    def __init__(self, C:int):
        super().__init__()
        self.conv = nn.Conv2d(C, C, 3, stride=1, padding=1)
    
    def forward(self, x:torch.Tensor):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=None, scale_factor=2, mode='nearest')
        x = self.conv(x)
        assert x.shape == (B, C, H*2, W*2), f"x.shape = ({B}, {C}, {H*2}, {W*2})"
        return x
    

class Nin(nn.Module):
    def __init__(self, in_dim, out_dim):
        '''
        :param C (int): number of input and output channels
        channel aggregation 1x1 conv와 매우 유사함
        (w,h)의 동일한 포인트에 있는 체널값들을 dot product로 곱하고 aggregation해서 output_channel을 만든다
        '''
        super().__init__()
        scale = 1e-10
        n = (in_dim + out_dim) / 2
        limit = np.sqrt(3*scale / n)

        # in/output dim에 맞춰서 uniform initialize 수행
        self.W = torch.nn.Parameter(torch.zeros((in_dim, out_dim), dtype=torch.float32).uniform_(-limit,+limit))
        self.b = torch.nn.Parameter(torch.zeros((1, out_dim, 1, 1), dtype=torch.float32))

    def forward(self, x):
        # 유사 1x1 실행
        return torch.einsum('bchw, co -> bowh', x, self.W) + self.b # dot product
    

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.dense = nn.Linear(512, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)

        if in_ch != out_ch:
            self.nin = Nin(in_ch, out_ch)
        
        self.dropout_rate = dropout_rate
    
    def forward(self, x, t_emb):
        '''
        t_emb.shape = (b, 512)
        h.shape = (b, h, c, w)
        '''
        # 노말라이즈, 엑티베이션, 컨벌루션
        h = F.group_norm(x, num_groups=32)
        h = F.silu(h)
        h = self.conv1(h)

        # 타임 임배딩 더하기
        t_emb = self.dense(F.silu(t_emb))[:, :, None, None]
        # print("line25", h.shape, t_emb.shape)
        h += t_emb

        # 노말라이즈, 액티베이션
        h = F.silu(F.group_norm(h, num_groups=32))
        # 드롭아웃
        h = F.dropout(h, p=self.dropout_rate)
        # 컨벌루션
        h = self.conv2(h)

        # channel 사이즈가 다르면 nin 모듈 통과 # nin은 1x1
        if x.shape[1] != h.shape[1]:
            x = self.nin(x)

        assert x.shape[1] == h.shape[1], f"x.shape[1] == h.shape[1], {x.shape[1]} == {h.shape[1]}"
        
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.ch = ch

        self.Q = Nin(ch, ch)
        self.K = Nin(ch, ch)
        self.V = Nin(ch, ch)

        self.nin = Nin(ch, ch)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.ch, f"c is {C}, self.ch is {self.ch}"

        h = F.group_norm(x, num_groups=32)
        
        q = self.Q(h) # b, ch, h, w
        k = self.K(h) # b, ch, h, w
        v = self.V(h) # b, ch, h, w

        w = torch.einsum('bchw, bcHW -> bhwHW', q, k) * (int(C) ** (-0.5)) # [B, H, W, H, W]
        w = torch.reshape(w, [B, H, W, H*W])
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, [B, H, W, H, W])

        h = torch.einsum('bhwHW, bcHW -> bchw', w, v)
        h = self.nin(h)
        assert h.shape == x.shape, f"h.shape == x.shape {h.shape} == {x.shape}"
        return x + h
    

class UNet(nn.Module):
    def __init__(self, ch=128, in_ch=1):
        super().__init__()
        self.ch = ch
        self.linear1 = nn.Linear(ch, 4*ch)
        self.linear2 = nn.Linear(4*ch, 4*ch)
        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride=1, padding=1)
        self.down = nn.ModuleList(
            [
                ResNetBlock(ch, 1*ch), # 0
                ResNetBlock(1*ch, 1*ch), # 1
                Downsample(1*ch), # 2

                ResNetBlock(1*ch, 2*ch), # 3
                AttentionBlock(2*ch), # 4
                ResNetBlock(2*ch, 2*ch), # 5
                AttentionBlock(2*ch), # 6
                Downsample(2*ch), # 7

                ResNetBlock(2*ch, 2*ch), # 8
                ResNetBlock(2*ch, 2*ch), # 9
                Downsample(2*ch), # 10
                ResNetBlock(2*ch, 2*ch), # 11
                ResNetBlock(2*ch, 2*ch), # 12
                ]
        )

        self.middle = nn.ModuleList(
            [
                ResNetBlock(2*ch, 2*ch),
                AttentionBlock(2*ch),
                ResNetBlock(2*ch, 2*ch),
            ]
            )
        
        self.up = nn.ModuleList(
            [
                ResNetBlock(4*ch, 2*ch), # 0
                ResNetBlock(4*ch, 2*ch), # 1
                ResNetBlock(4*ch, 2*ch), # 2
                Upsample(2*ch), # 3

                ResNetBlock(4*ch, 2*ch), # 4
                ResNetBlock(4*ch, 2*ch), # 5
                ResNetBlock(4*ch, 2*ch), # 6
                Upsample(2*ch), # 7

                ResNetBlock(4*ch, 2*ch), # 8
                AttentionBlock(2*ch), # 9
                ResNetBlock(4*ch, 2*ch), # 10
                AttentionBlock(2*ch), # 11
                ResNetBlock(3*ch, 2*ch), # 12
                AttentionBlock(2*ch), # 13

                Upsample(2*ch), # 14
                ResNetBlock(3*ch, ch), # 15
                ResNetBlock(2*ch, ch), # 16
                ResNetBlock(2*ch, ch), # 17
            ]
        )

        self.final_conv = nn.Conv2d(ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, t):
        '''
        :param x: batch image tensor [B, C, H, W]
        :param t: time embed [B] dtype torch.long
        '''
        temb = get_timestep_embedding(t, self.ch) # [b, ch]
        temb = self.linear1(temb) # [b, 4*ch]
        temb = self.linear2(temb) # [b, 4*ch]

        assert temb.shape == (t.shape[0], 4*self.ch), f"temb.shape == (t, 4*self.ch), {temb.shape} == ({t.shape[0]}, 4*{self.ch})"
        
        x1 = self.conv1(x)

        # Down
        x2 = self.down[0](x1, temb) # 64x64
        x3 = self.down[1](x2, temb) # 64x64
        x4 = self.down[2](x3) # 32x32
        x5 = self.down[3](x4, temb) # 32x32
        x6 = self.down[4](x5) # 32x32 Attn
        x7 = self.down[5](x6, temb) # 32x32
        x8 = self.down[6](x7) # 32x32 Attn
        x9 = self.down[7](x8) # 16x16
        x10 = self.down[8](x9, temb) # 16x16
        x11 = self.down[9](x10, temb) # 16x16
        x12 = self.down[10](x11) # 8x8
        x13 = self.down[11](x12, temb) # 8x8
        x14 = self.down[12](x13, temb) # 8x8
        
        # Middle
        x = self.middle[0](x14, temb) # 8x8
        x = self.middle[1](x) # 8x8
        x = self.middle[2](x, temb) # 8x8

        # Up
        x = self.up[0](torch.cat((x, x14), dim=1), temb)
        x = self.up[1](torch.cat((x, x13), dim=1), temb)
        x = self.up[2](torch.cat((x, x12), dim=1), temb)
        x = self.up[3](x)
        x = self.up[4](torch.cat((x, x11), dim=1), temb)
        x = self.up[5](torch.cat((x, x10), dim=1), temb)
        x = self.up[6](torch.cat((x, x9), dim=1), temb)
        x = self.up[7](x)
        x = self.up[8](torch.cat((x, x8), dim=1), temb)
        x = self.up[9](x)
        x = self.up[10](torch.cat((x, x6), dim=1), temb)
        x = self.up[11](x)
        x = self.up[12](torch.cat((x, x4), dim=1), temb)
        x = self.up[13](x)
        x = self.up[14](x)
        x = self.up[15](torch.cat((x, x3), dim=1), temb)
        x = self.up[16](torch.cat((x, x2), dim=1), temb)
        x = self.up[17](torch.cat((x, x1), dim=1), temb)

        x = F.silu(F.group_norm(x, num_groups=32))
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    unet = UNet(128)
    t = torch.Tensor(4)
    x = torch.Tensor(4, 1, 32, 32)
    out = unet(x, t)
    print(out.shape) # (original_resolution//8, original_resolution//8)