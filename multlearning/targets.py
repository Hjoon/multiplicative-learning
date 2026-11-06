"""Equations ported from ER_abs_vs_difference.ipynb, cell 3.

DTP corrects BOTH the hidden absolute inverse and the absolute affine weight
reconstruction, exactly as that notebook. We do not call atanh: ER's stabilized
secant target S*=S+g(S)(Z*-Z) is preserved.
"""
import torch
from .er import stable_secant_gain

class Inverse:
    """Fixed linear inverse; fuse RHS/reuse factors without changing the solve."""
    def __init__(self,A,cfg):
        self.A=A; self.dtype=A.dtype;self.work=A.to(getattr(torch,cfg.solve_dtype));self.cfg=cfg
        n,d=A.shape;self.primal=d<=n if cfg.ridge_strategy=='auto' else cfg.ridge_strategy=='primal'
        self.info=0
        if cfg.er_solver=='pinv':self.pinv=torch.linalg.pinv(self.work);return
        if cfg.er_solver!='ridge':raise ValueError(cfg.er_solver)
        lam=cfg.ridge_lambda
        if cfg.ridge_relative:lam=lam*(self.work.square().sum()/d).clamp_min(torch.finfo(self.work.dtype).eps)
        gram=self.work.T@self.work if self.primal else self.work@self.work.T
        gram.diagonal().add_(lam)
        self.chol,self.info=torch.linalg.cholesky_ex(gram,check_errors=False)
    def __call__(self,B):
        B=B.to(self.work.dtype)
        if self.cfg.er_solver=='pinv':X=self.pinv@B
        elif self.primal:X=torch.cholesky_solve(self.work.T@B,self.chol)
        else:X=self.work.T@torch.cholesky_solve(B,self.chol)
        return X.to(self.dtype)

class AffineFit:
    def __init__(self,Z,cfg):
        self.Z=Z;self.mean=Z.mean(0,keepdim=True)
        self.operator=Inverse(Z-self.mean,cfg)
    def __call__(self,S):
        mean=S.mean(0,keepdim=True);W=self.operator(S-mean)
        b=(mean-self.mean@W).squeeze(0)
        return W,b


def inverse_targets(Z,S,Sstar,weight,bias,cfg,propagation):
    if propagation not in ('tp','dtp'):raise ValueError(propagation)
    op=Inverse(weight,cfg)
    # Same inverse in both arms; target/current RHS fused into one factor solve.
    both=op(torch.cat(((Sstar-bias).T,(S-bias).T),dim=1)).T
    inv_target,inv_current=both.split(len(S),dim=0)
    target=inv_target if propagation=='tp' else inv_target-(inv_current-Z)
    residual=(inv_current-Z).norm()/Z.norm().clamp_min(1e-12)
    return target,residual


def weight_candidate(fit,Sstar,S,oldW,oldb,propagation):
    if propagation not in ('tp','dtp'):raise ValueError(propagation)
    W,b=fit(torch.cat((Sstar,S),dim=1));out=S.shape[1]
    Wt,Wr=W.split(out,dim=1);bt,br=b.split(out,dim=0)
    return (Wt,bt) if propagation=='tp' else (Wt-(Wr-oldW),bt-(br-oldb))


@torch.no_grad()
def step(model,x,target,cfg,propagation,first_fit=None,diagnostics=False):
    z,s=model.forward_with_cache(x);L=len(s);zt=[None]*(L+1);st=[None]*L;zt[L]=target
    residuals={}
    for l in range(L-1,-1,-1):
        st[l]=s[l]+stable_secant_gain(s[l],clip=cfg.secant_clip,small=cfg.secant_small,
                       use_small_series=cfg.secant_small_series)*(zt[l+1]-z[l+1])
        if l>0:
            zt[l],r=inverse_targets(z[l],s[l],st[l],model.layers[l].weight,
                                    model.layers[l].bias[None,:],cfg,propagation)
            if diagnostics:residuals[f'inverse_reconstruction_residual_layer_{l+1}']=float(r)
    current=x
    for l,layer in enumerate(model.layers):
        fit=first_fit if l==0 and first_fit is not None else AffineFit(current,cfg)
        W,b=weight_candidate(fit,st[l],s[l],layer.weight.T,layer.bias,propagation)
        layer.weight.add_(cfg.er_mix*(W.T-layer.weight));layer.bias.add_(cfg.er_mix*(b-layer.bias))
        current=torch.tanh(layer(current))
    if diagnostics:
        for l,layer in enumerate(model.layers):residuals[f'weight_norm_layer_{l+1}']=float(layer.weight.norm())
    return residuals
