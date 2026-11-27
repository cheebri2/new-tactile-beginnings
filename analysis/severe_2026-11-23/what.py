import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from piezo import locate
from locator import stamps
tr,fe=locate.load('/mnt/user-data/uploads/final_data','cw90',['dl'])
h=stamps(tr,['healthy_vertebra']); s=stamps(tr,['severe_scoliosis'])
Xh,_,ch,fr=locate.share_matrix(fe,h.trial_id,f_max=200); Xs,*_=locate.share_matrix(fe,s.trial_id,ch,fr)
xs,ys=locate.mold('severe'); o=np.argsort(ys); S=dict(zip([0,30,60,85,125],xs[o]))
Sv=s.y_mm.map(S).values; x=s.x_mm.values.astype(float); g=s.y_mm.values
# combined healthy+severe, leave-one-severe-row-out; targets: head vs mold (x) or head vs spine (x-S)
gh=np.array(['h%g'%v for v in h.y_mm])
for name,ys_ in [('head vs mold centreline (x)',x),('head vs spine (x - S)',x-Sv)]:
    p=np.zeros(len(x))
    for r in np.unique(g):
        t=g!=r; Xtr=np.vstack([Xh,Xs[t]]); ytr=np.r_[h.x_mm.values,ys_[t]]; gtr=np.r_[gh,g[t].astype(str)]
        m=locate.ridge_fit(Xtr,ytr,locate.pick_alpha(Xtr,ytr,gtr)); p[~t]=locate.ridge_predict(m,Xs[~t])
    sl=np.polyfit(ys_,p,1)[0]; r_=np.corrcoef(p,ys_)[0,1]
    print(f'{name:30s} held-out-row MAE {np.mean(abs(p-ys_)):.2f}  corr {r_:.2f} slope {sl:.2f}  baseline {np.mean(abs(ys_-ys_.mean())):.2f}')
# severe only, leave-one-row-out
for name,ys_ in [('severe only: x',x),('severe only: x - S',x-Sv)]:
    p=locate.grouped_cv(Xs,ys_,g); print(f'{name:30s} MAE {np.mean(abs(p-ys_)):.2f} corr {np.corrcoef(p,ys_)[0,1]:.2f} baseline {np.mean(abs(ys_-ys_.mean())):.2f}')
# partial: regress healthy-model prediction on x and S jointly
raw=locate.grouped_cv(Xh,h.x_mm.values.astype(float),h.y_mm.values); sl,ic=np.polyfit(h.x_mm.values,raw,1)
m=locate.ridge_fit(Xh,h.x_mm.values.astype(float),10.0); ph=(locate.ridge_predict(m,Xs)-ic)/sl
A=np.c_[x,Sv,np.ones_like(x)]; coef,*_=np.linalg.lstsq(A,ph,rcond=None); print('healthy-model pred = %.2f*x + %.2f*S + %.2f  (spine-reading would be 1, -1)'%tuple(coef))
# with row dummies? no: S constant per row so use only that
