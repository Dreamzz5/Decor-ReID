from torch import nn

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1.permute(0, 2, 1) if x2 is None else x2.permute(0, 2, 1)
    w1 = x1.norm(p=2, dim=-1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=-2, keepdim=True)
    return torch.bmm(x1, x2) / (w1 * w2).clamp(min=eps)

class EMA_FM():
    def __init__(self, decay=0.9, first_decay=0.0, channel_size=512, f_map_size=196, is_use = False):
        self.decay = decay
        self.first_decay = first_decay
        self.is_use = is_use
        self.shadow = {}
        self.epsional = 1e-5
        if is_use:
            self._register(channel_size=channel_size, f_map_size= f_map_size)

    def _register(self, channel_size=512, f_map_size=196):
        Init_FM = torch.zeros((f_map_size, channel_size),dtype=torch.float)
        self.shadow['FM'] = Init_FM.clone()
        self.is_first = True

    def update(self, input):
        B, C, _ = input.size()
        if not(self.is_use):
            return torch.ones((C,C), dtype=torch.float)
        decay = self.first_decay if self.is_first else self.decay
        ####### FEATURE SIMILARITY MATRIX EMA ########
        # Mu = torch.mean(input,dim=0)
        self.shadow['FM'] = (1.0 - decay) * torch.mean(input,dim=0) + decay * self.shadow['FM'].to(input)
        self.is_first = False
        return self.shadow['FM']

class SMGBlock(nn.Module):
    def __init__(self, channel_size=2048, f_map_size=196):
        super(SMGBlock, self).__init__()

        self.EMA_FM = EMA_FM(decay=0.95, first_decay=0.0, channel_size=channel_size, f_map_size=f_map_size, is_use=True)


    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        batch_size, channel, _,_ = x.size()
        theta_x = x.view(batch_size, channel, -1).permute(0, 2, 1).contiguous()
        with torch.no_grad():
            f_mean = self.EMA_FM.update(theta_x)
        sz = f_mean.size()[0]
        f_mean = f_mean.view(1,channel,sz)
        Local = theta_x.permute(0, 2, 1)-f_mean
        similarity = cosine_similarity_torch(Local)
        return similarity


if __name__ == '__main__':
    import torch

    img = torch.rand(1, 1024, 14, 14).cuda()
    net = SMGBlock(1024, 196)
    out = net(img)
    print(out.size())



