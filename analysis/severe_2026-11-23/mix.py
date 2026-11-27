import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from piezo import locate
from locator import stamps
from cobb import fit_segments, fit_c
tr,fe=locate.load('/mnt/user-data/uploads/final_data','cw90',['dl'])
h=stamps(tr,['healthy_vertebra']); s=stamps(tr,['severe_scoliosis'])
fmax=float(sys.argv[1]) if len(sys.argv)>1 else 200
Xh,_,ch,fr=locate.share_matrix(fe,h.trial_id,f_max=fmax); Xs,*_=locate.share_matrix(fe,s.trial_id,ch,fr)
xs,ys=locate.mold('severe'); o=np.argsort(ys); xs,ys=xs[o],ys[o]; rows=[0,30,60,85,125]; S=dict(zip(rows,xs))
yh=h.x_mm.values.astype(float); yl=(s.x_mm-s.y_mm.map(S)).values; gs=s.y_mm.values; rs=s.repeat.values
def fitpred(Xtr,ytr,gtr,Xte):
    a=locate.pick_alpha(Xtr,ytr,gtr); m=locate.ridge_fit(Xtr,ytr,a); return locate.ridge_predict(m,Xte)
def cal(Xtr,ytr,gtr):  # calibration from grouped CV on training set
    raw=locate.grouped_cv(Xtr,ytr,gtr); sl,ic=np.polyfit(ytr,raw,1); return sl,ic
def report(name,pred):
    e=pred-yl; sp=s.x_mm.values-pred
    X=pd.Series(sp).groupby(gs).median().reindex(rows).values
    print(f'{name:44s} MAE {np.mean(abs(e)):.2f} by row',pd.Series(abs(e)).groupby(gs).mean().round(1).to_dict(),
          '| spine',X.round(1),'| Cobb seg %.1f c %.1f (design 47.4/54.8)'%(fit_segments(ys,X)[0],fit_c(ys,X)[0]))
    return e
gh=np.array(['h%g'%v for v in h.y_mm])
# 1. healthy only
sl,ic=cal(Xh,yh,gh); report('healthy only',(fitpred(Xh,yh,gh,Xs)-ic)/sl)
# 2. healthy + other severe rows, hold out one severe row
p=np.zeros(len(yl)); p_nocal=np.zeros(len(yl))
for r in rows:
    tr_=gs!=r; Xtr=np.vstack([Xh,Xs[tr_]]); ytr=np.r_[yh,yl[tr_]]; gtr=np.r_[gh,['s%g'%v for v in gs[tr_]]]
    raw=fitpred(Xtr,ytr,gtr,Xs[~tr_]); p_nocal[~tr_]=raw
    sl,ic=cal(Xtr,ytr,gtr); p[~tr_]=(raw-ic)/sl
report('healthy + 4 severe rows, test 5th (no cal)',p_nocal); report('healthy + 4 severe rows, test 5th (cal)',p)
# 3. half split by reseat: train reps 1 (+2?) ... bri's 1/2: train on r01 + half of r02? use r1,r2 train r3 test, and r3 train r1 test
for trr,ter in [([1,2],[3]),([3],[1,2]),([1],[2,3])]:
    tr_=np.isin(rs,trr); te=np.isin(rs,ter)
    Xtr=np.vstack([Xh,Xs[tr_]]); ytr=np.r_[yh,yl[tr_]]; gtr=np.r_[gh,['s%g'%v for v in gs[tr_]]]
    raw=fitpred(Xtr,ytr,gtr,Xs[te]); sl,ic=cal(Xtr,ytr,gtr); pp=(raw-ic)/sl
    e=pp-yl[te]; sp=s.x_mm.values[te]-pp; X=pd.Series(sp).groupby(gs[te]).median().reindex(rows).values
    print(f'train reseats {trr} test {ter}: MAE {np.mean(abs(e)):.2f}  spine {X.round(1)}  Cobb seg {fit_segments(ys,X)[0]:.1f}')
