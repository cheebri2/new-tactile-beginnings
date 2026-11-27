import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from piezo import locate
from locator import stamps
tr,fe=locate.load('/mnt/user-data/uploads/final_data','cw90',['dl'])
h=stamps(tr,['healthy_vertebra']); s=stamps(tr,['severe_scoliosis'])
Xh,_,ch,fr=locate.share_matrix(fe,h.trial_id,f_max=200); Xs,*_=locate.share_matrix(fe,s.trial_id,ch,fr)
gh=h.y_mm.values
raw=locate.grouped_cv(Xh,h.x_mm.values.astype(float),gh); sl,ic=np.polyfit(h.x_mm.values,raw,1)
m=locate.ridge_fit(Xh,h.x_mm.values.astype(float),locate.pick_alpha(Xh,h.x_mm.values.astype(float),gh))
p=(locate.ridge_predict(m,Xs)-ic)/sl
xs,ys=locate.mold('severe')
for sig in [5,10,15,20,25,30,40,60]:
    c=s.y_mm.values+47
    w=np.exp(-(ys[None,:]-c[:,None])**2/(2*sig**2)); w/=w.sum(1,keepdims=True)
    eff=w@xs; lab=s.x_mm.values-eff; e=p-lab
    print(sig,'MAE %.2f'%np.mean(abs(e)),'row mean err',pd.Series(e).groupby(s.y_mm.values).mean().round(1).to_dict(),'eff',pd.Series(eff).groupby(s.y_mm.values).first().round(1).to_dict())
