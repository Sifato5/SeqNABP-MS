import torch
from torch import nn
class ResConv(nn.Module):
    def __init__(self,d,k=3,dil=1,drop=.2):
        super().__init__(); self.conv=nn.Conv1d(d,d,k,padding=dil*(k-1)//2,dilation=dil)
        self.norm=nn.BatchNorm1d(d); self.gate=nn.Conv1d(d,2*d,1); self.drop=nn.Dropout(drop)
    def forward(self,x):
        z=self.norm(self.conv(x.transpose(1,2)))
        a,b=self.gate(z).chunk(2,1); z=(a.sigmoid()*b.tanh()).transpose(1,2)
        return x+self.drop(z)

class MambaSequenceBlock(nn.Module):
    """Residual Mamba block. Requires the official mamba-ssm package."""
    def __init__(self,d,drop):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError(
                "Could not import mamba_ssm. Install a CUDA/PyTorch-compatible "
                "mamba-ssm build; see README for environment-specific installation."
            ) from e
        self.norm=nn.LayerNorm(d)
        self.mamba=Mamba(d_model=d,d_state=16,d_conv=4,expand=2)
        self.drop=nn.Dropout(drop)
    def forward(self,x,mask):
        y=self.mamba(self.norm(x))
        return (x+self.drop(y))*mask.unsqueeze(-1)

class AttnPool(nn.Module):
    def __init__(self,d):
        super().__init__(); self.a=nn.Sequential(nn.Linear(d,d//2),nn.Tanh(),nn.Linear(d//2,1))
    def forward(self,x,m):
        w=self.a(x).squeeze(-1).masked_fill(~m,-1e9).softmax(1).unsqueeze(-1)
        mu=(w*x).sum(1); sd=((w*(x-mu[:,None])**2).sum(1).clamp_min(1e-6)).sqrt()
        mean=(x*m[:,:,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
        mx=x.masked_fill(~m[:,:,None],-1e9).max(1).values
        return torch.cat([mu,sd,mean,mx],-1)
class SeqNABPMS(nn.Module):
    def __init__(self,input_dim=2560,d=256,heads=8,drop=.3):
        super().__init__(); self.proj=nn.Sequential(nn.Linear(input_dim,d),nn.LayerNorm(d))
        self.convs=nn.ModuleList([ResConv(d,3,1,drop),ResConv(d,5,1,drop),ResConv(d,3,2,drop),ResConv(d,3,4,drop)])
        self.branch=nn.Sequential(nn.Linear(4*d,d),nn.Sigmoid())
        self.mamba=MambaSequenceBlock(d,drop)
        layer=nn.TransformerEncoderLayer(d,heads,4*d,drop,batch_first=True,norm_first=True,activation="gelu")
        self.transformer=nn.TransformerEncoder(layer,2)
        self.fuse=nn.Sequential(nn.Linear(2*d,d),nn.Sigmoid()); self.pool=AttnPool(d)
        self.shared=nn.Sequential(nn.Linear(4*d,d*2),nn.GELU(),nn.Dropout(drop),nn.Linear(d*2,d),nn.GELU())
        self.cls=nn.Linear(d,4); self.nabp=nn.Linear(d,2); self.dbr=nn.Linear(d,2); self.ssd=nn.Linear(d,2)
    def forward(self,x,m):
        h=self.proj(x)*m[:,:,None]; bs=[b(h)*m[:,:,None] for b in self.convs]
        local=sum(bs)*self.branch(torch.cat(bs,-1))
        s=self.mamba(local,m); a=self.transformer(local,src_key_padding_mask=~m)
        g=self.fuse(torch.cat([s,a],-1)); z=self.shared(self.pool(g*s+(1-g)*a,m))
        return {"logits":self.cls(z),"nabp":self.nabp(z),"dbr":self.dbr(z),"ssd":self.ssd(z)}
