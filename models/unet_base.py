import torch
import torch.nn as nn

'''
- 다운 블록
- 미드 블록
- 업 블록
- 타임 임배딩 블록

- 다운 블록
-- 레즈넷 블록
-- 셀프어텐션 블록
-- |레즈_블록 +  타임임배딩 + 레즈_블록| + |셀프어텐션| + 다운샘플
-- 타임 임배딩은 브로드캐스팅을 해서 정보주입

'''

# t(B,) -> t_emb(B, t_emb_dim) # Batch는 time_steps의 개수
def get_time_embedding(time_steps, t_emb_dim):
    factor = 10_000 ** ((torch.arange(
        start=0, end=t_emb_dim//2, device=time_steps.device) / (t_emb_dim//2)
        ))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim//2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels),
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4,
                                          stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        '''
        x.shape = (b, c, h, w)  
        t_emb.shape = (b, c)  
        out.shape = (b, c, h, w) or (b, c, h//2, w//2)  
        '''
        out = x
        # Resnet Block
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None] # (b, c, h, w) + (b, c, 1, 1)
        out = self.resnet_conv_second(out) 
        out = out + self.residual_input_conv(resnet_input) # (b, c, h, w)
        
        # Attention block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w) # (b, c, hw)
        in_attn = self.attention_norm(in_attn) # (b, c, hw)
        in_attn = in_attn.transpose(1, 2) # (b, hw, c)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn) # (b, hw, c)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w) # (b, hw, c) -> (b, c, hw) -> (b, c, h, w)
        out = out + out_attn

        out = self.down_sample_conv(out)
        return out

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
        ])
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels),
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels),
            )
        ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ),
        ])
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        ])
    
    def forward(self, x, t_emb):
        out = x
        # first resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # attention block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        # second resnet block
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up_sample_conv = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=4,
                                                 stride=2, padding=1) if self.up_sample else nn.Identity()
        
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)

        # Resnet Block
        out = x
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention Block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        return out


class Unet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1): # 0,1,2
            self.downs.append(
                DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                          down_sample=self.down_sample[i], num_heads=4)
                          )
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1): # 0, 1
            self.mids.append(
                MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim, num_heads=4)
                )
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(
                UpBlock(self.down_channels[i]*2, self.down_channels[i-1] if i!=0 else 16,
                        self.t_emb_dim, up_sample=self.down_sample[i], num_heads=4)
                )
            
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        out = self.conv_in(x)
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        for down in self.downs:
            # print(out.shape)
            down_outs.append(out)
            out= down(out,t_emb)
        
        for mid in self.mids:
            # print(out.shape)
            out = mid(out, t_emb)
        
        for up in self.ups:
            down_out = down_outs.pop()
            # print(out.shape, down_out.shape)
            out = up(out, down_out, t_emb)
        
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out



if __name__ == '__main__':
    unet = Unet(1)
    img = torch.Tensor(64, 1, 28, 28)
    out = unet.forward(img, torch.Tensor(64))
    print(out.shape)
