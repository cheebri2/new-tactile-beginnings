import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from piezo import locate
from locator import stamps
from cobb import fit_segments
tr,fe=locate.load('/mnt/user-data/uploads/final_data','cw90',['dl'])
h=stamps(tr,['healthy_vertebra']); s=stamps(tr,['severe_scoliosis'])
fe=fe[(fe.freq_hz>=50)&(fe.freq_hz<=400)]
chans=['dl','dr','ur']; freqs=sorted(fe.freq_hz.unique())
P=fe.pivot_table(index='trial_id',columns=['channel','freq_hz'],values=['phase_deg','amp'])
def feat(ids,kind,fmax=400):
    F=[f for f in freqs if f<=fmax]; out=[]
    ph=np.stack([P['phase_deg'][c][F].loc[ids].values for c in chans],1)   # trial x ch x f
    la=np.log10(np.stack([P['amp'][c][F].loc[ids].values for c in chans],1).clip(1e-3))
    if kind=='share amp': out=[(la-la.mean(1,keepdims=True)).reshape(len(ids),-1)]
    if kind in('phase vs drive','phase both'):
        z=np.exp(1j*np.radians(ph)); out+=[z.real.reshape(len(ids),-1),z.imag.reshape(len(ids),-1)]
    if kind in('phase between discs','phase both'):
        rel=np.angle(np.exp(1j*np.radians(ph-ph[:,[2]])))[:,:2]   # vs ur (same as recorded ul? phantom ur)
        out+=[np.cos(rel).reshape(len(ids),-1),np.sin(rel).reshape(len(ids),-1)]
    if kind=='peak frequency':
        # resonance: freq of max amp per disc (parabolic interp) + phase slope (group delay) per disc
        A=10**la; fa=np.array(F); pk=[]
        for c in range(3):
            i=A[:,c].argmax(1); pk.append(fa[i])
            unw=np.unwrap(np.radians(ph[:,c]),axis=1); pk.append(np.polyfit(fa,unw.T,1)[0])
        out=[np.array(pk).T]
    return np.hstack(out)
xs,ys=locate.mold('severe'); o=np.argsort(ys); S=dict(zip([0,30,60,85,125],xs[o])); Yl=ys[o]
yh=h.x_mm.values.astype(float); gh=h.y_mm.values; x=s.x_mm.values.astype(float); Sv=s.y_mm.map(S).values
for fmax in ():
  for kind in []:
    Xh=feat(h.trial_id,kind,fmax); Xs=feat(s.trial_id,kind,fmax)
    raw=locate.grouped_cv(Xh,yh,gh); sl,ic=np.polyfit(yh,raw,1)
    m=locate.ridge_fit(Xh,yh,locate.pick_alpha(Xh,yh,gh)); p=(locate.ridge_predict(m,Xs)-ic)/sl
    cv=(raw-ic)/sl
    A=np.c_[x,Sv,np.ones_like(x)]; co,*_=np.linalg.lstsq(A,p,rcond=None)
    sp=pd.Series(x-p).groupby(s.y_mm.values).median().values
    print(f'f<={fmax} {kind:20s} healthy CV MAE {np.mean(abs(raw-yh)):.2f} (slope {sl:.2f}) | severe MAE {np.mean(abs(p-(x-Sv))):.2f} | pred = {co[0]:.2f}x {co[1]:+.2f}S | spine {np.round(sp,1)} Cobb {fit_segments(Yl,sp)[0]:.1f}')
rng=np.random.default_rng(0)
print('--- screening: stamps only on the centreline (x=0)')
hy=[0,20,60,80,100]; hY=np.array([38.6,70.3,98.2,122.3,152.2]); sy=[0,30,60,85,125]
def auc(a,b): return np.mean([(bb>aa)+0.5*(bb==aa) for aa in a for bb in b])
for kind,fmax in [('share amp',200),('phase between discs',200),('phase between discs',400)]:
    Xh=feat(h.trial_id,kind,fmax); Xs=feat(s.trial_id,kind,fmax)
    raw=locate.grouped_cv(Xh,yh,gh); sl,ic=np.polyfit(yh,raw,1); ph=(raw-ic)/sl     # held-out-row on healthy
    m=locate.ridge_fit(Xh,yh,locate.pick_alpha(Xh,yh,gh)); ps=(locate.ridge_predict(m,Xs)-ic)/sl
    H={r:-ph[(h.y_mm.values==r)&(yh==0)] for r in hy}; Sv_={r:-ps[(s.y_mm.values==r)&(x==0)] for r in sy}
    for n in (1,3):
        def draw(D,keys,Y):
            v=np.array([np.median(rng.choice(D[k],n,replace=False)) for k in keys])
            return fit_segments(Y,v)[0], np.std(v)
        a=np.array([draw(H,hy,hY) for _ in range(2000)]); b=np.array([draw(Sv_,sy,Yl) for _ in range(2000)])
        for j,stat in enumerate(['Cobb','spread SD']):
            thr=np.median(np.r_[a[:,j],b[:,j]])
            print(f'{kind:20s} f<={fmax} {n} stamp/vertebra {stat:9s}: healthy {a[:,j].mean():.1f}±{a[:,j].std():.1f}  severe {b[:,j].mean():.1f}±{b[:,j].std():.1f}  AUC {auc(a[:300,j],b[:300,j]):.2f}  healthy>=10deg {100*np.mean(a[:,0]>=10):.0f}% severe>=10deg {100*np.mean(b[:,0]>=10):.0f}%' if j==0 else f'{"":44s}{stat:9s}: healthy {a[:,j].mean():.1f}±{a[:,j].std():.1f}  severe {b[:,j].mean():.1f}±{b[:,j].std():.1f}  AUC {auc(a[:300,j],b[:300,j]):.2f}')
