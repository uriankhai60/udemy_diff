import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm

def sample_batch(batch_size=120, device='cpu'):
    data, _ = make_swiss_roll(batch_size)
    data = data[: ,[2, 0]] / 10 # swap X~Y
    data = data * np.array([1, -1]) # flip
    return torch.from_numpy(data).to(device)


def plot(model, file_name, device):
    fontsize=14
    fig = plt.figure(figsize=(10, 6))

    N=5_000
    x0 = sample_batch(N).to(device)
    samples = model.sample(N, device)

    data = [
        x0.cpu(), 
        model.forward_process(x0, 20)[-1].cpu(), 
        model.forward_process(x0, 40)[-1].cpu()
        ]
    
    for i in range(3):
        plt.subplot(2, 3, 1+i)
        plt.scatter(data[i][:, 0].data.numpy(), data[i][:, 1].data.numpy(), alpha=0.1, s=1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if i == 0: plt.ylabel(r'$q(\mathbf{x}^{(0..T)})$', fontsize=fontsize)
        if i == 0: plt.title(r'$t=0$', fontsize=fontsize)
        if i == 1: plt.title(r'$t=\frac{T}{2}$', fontsize=fontsize)
        if i == 2: plt.title(r'$t=T$', fontsize=fontsize)
    
    time_steps = [0, 20, 40]
    for i in range(3):
        plt.subplot(2, 3, 4+i)
        plt.scatter(samples[time_steps[i]][:, 0].data.cpu().numpy(), samples[time_steps[i]][:, 1].data.cpu().numpy(), 
                    alpha=0.1, c='r', s=1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')

        if i == 0: plt.ylabel(r'$p(\mathbf{x}^{(0..T)})$', fontsize=fontsize)

    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


class MLP(nn.Module):
    def __init__(self, N=40, data_dim=2, hidden_dim=64):
        super().__init__()
        self.network_head = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.network_tail = nn.ModuleList(
            [ 
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, data_dim * 2),
                )
                for i in range(N)
            ]
        )
    
    def forward(self, x, t):
        h = self.network_head(x)
        tmp = self.network_tail[t](h) # b, d*2
        mu, h = torch.chunk(tmp, 2, dim=1) # b,d b,d
        var = torch.exp(h)
        std = torch.sqrt(var)
        return mu, std


class DiffusionModel():

    def __init__(self,
                 T,
                 model:nn.Module,
                 device,
                 dim=2):
        self.betas = (
            torch.sigmoid(torch.linspace(-18, 10, T)) * (3e-1 - 1e-5) + 1e-5
            ) # 1e-5 ~ 0.3까지 균일하게 나눈 리스트
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)

        self.T = T
        self.model = model
        self.dim = dim
        self.device = device

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)

    def forward_process(self, x0, t):
        '''
        input: x0, t
        output: xt 평균(t), xt 표준편차(t), xt
        
        '''
        assert t>0, f'forward_process, t>0'
        assert t<=self.T, f'forward_process, t<=self.T'

        t = t-1 # because we start indexing at 0

        mu = torch.sqrt(self.alphas_bar[t]) * x0
        std = torch.sqrt(1 - self.alphas_bar[t])
        epsilon = torch.randn_like(x0)

        xt = mu + epsilon * std # data ~ N(mu, std)

        ## forward때의 분포(= 가우시안 분포를 가정)
        ## 가우시안 분포의 std, mu는 베타 스케줄에 의해서 fixed 
        
        # 초록색 박스
        std_q = torch.sqrt(
            (1-self.alphas_bar[t-1]) / (1-self.alphas_bar[t]) * self.betas[t]
        ) 
        
        # 파랑색 박스
        m1 = torch.sqrt(self.alphas_bar[t-1]) * self.betas[t] / (1 - self.alphas_bar[t])
        m2 = torch.sqrt(self.alphas_bar[t]) * (1 - self.alphas_bar[t-1]) / (1 - self.alphas_bar[t])
        mu_q = m1*x0 + m2*xt

        return mu_q, std_q, xt
    
    def reverse_process(self, xt, t):
        '''
        input: x_noised, t
        output: t스텝 x의 평균, t스텝 x의 표준편차, t-1스텝의 샘플링값
        '''
        assert t>0, 'reverse_process t>0'
        assert t<=self.T, 'reverse_process t<=self.T'
        
        t = t-1 # 스케줄의 인덱스는 0부터 시작하므로 맞추기 위함

        mu_p, std_p = self.model(xt, t)
        epsilon = torch.randn_like(xt)

        return mu_p, std_p, mu_p + epsilon*std_p # data ~ N(mu_p, std_p)
    


    def sample(self, batch_size, device):
        noise = torch.randn((batch_size, self.dim)).to(device)
        x = noise

        samples = [x] # at T

        for t in range(self.T, 0, -1):
            if t != 1:
                _, _, x = self.reverse_process(x, t)
            samples.append(x) # T-1, T-2, ..., 1, 0
        
        return samples[::-1] # 0, 1, 2, ..., T
            


    def get_loss(self, x0):
        '''
        input: x0
        output: loss
        '''
        # t스텝을 랜덤으로 선택
        t = torch.randint(2, 40+1, (1,)) 

        # t스텝의 평균, 표준편차, 샘플값
        mu_q, sigma_q, xt = self.forward_process(x0, t)

        # 예측한 t스텝의 평균, 표준편차, 샘플값
        mu_p, sigma_p, xt_minus1 = self.reverse_process(xt.float(), t)

        # KL 다이버전스, p와 q가 바뀌어서 봐야함
        KL = torch.log(sigma_p/sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2)/(2*sigma_p**2)
        K = -KL.mean()
        loss = -K
        
        return loss


def train(diffusion_model, optimizer, batch_size, nb_epochs, device):
    training_loss = []
    for epoch in tqdm(range(nb_epochs), leave=True):
        x0 = sample_batch(batch_size).to(device)
        loss = diffusion_model.get_loss(x0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())

        if epoch % 5000 == 0:
            plt.plot(training_loss)
            plt.savefig(f'figs_script/training_loss_epoch_{epoch}.png')
            plt.close()

            plot(diffusion_model, f'figs_script/training_epoch_{epoch}.png', device)
    
    print("Done!")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp_model = MLP(hidden_dim=128).to(device)
    diff_model = DiffusionModel(40, mlp_model, device, 2)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-4)
    train(diff_model, optimizer, 64_000, 300_000, device)