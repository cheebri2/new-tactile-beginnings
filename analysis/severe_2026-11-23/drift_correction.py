import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from piezo import locate
from locator import stamps
from cobb import fit_segments, fit_c
tr,fe=locate.load('/mnt/user-data/uploads/final_data','cw90',['dl'])
tr['t']=pd.to_datetime(tr.time); tr['hrs']=(tr.t-tr.t.min()).dt.total_seconds()/3600
chans=['dl','dr','ur']; F=[f for f in sorted(fe.freq_hz.unique()) if 50<=f<=200]
P=fe.pivot_table(index='trial_id',columns=['channel','freq_hz'],values='phase_deg')
def relang(ids):
    ph=np.stack([P[c][F].loc[ids].values for c in chans],1); return np.radians(ph-ph[:,[2]])[:,:2].reshape(len(ids),-1)
def toX(a): return np.hstack([np.cos(a),np.sin(a)])
def corrected(st,mode):
    a=relang(st.trial_id)
    if mode=='none': return toX(a)
    if mode in ('session-mean','stamp-trend'):
        out=a.copy()
        for sess in st.session.unique():
            m=(st.session==sess).values; z=np.exp(1j*a[m])
            if mode=='session-mean': out[m]=np.angle(z*np.conj(z.mean(0)))
            else:
                t=tr.set_index('trial_id').loc[st.trial_id[m],'hrs'].values
                base=np.angle(np.stack([np.polyval(np.polyfit(t,z[:,k].real,1),t)+1j*np.polyval(np.polyfit(t,z[:,k].imag,1),t) for k in range(z.shape[1])],1))
                out[m]=a[m]-base
        return toX(out)
    out=a.copy()
    for sess in st.session.unique():
        b=tr[(tr.session==sess)&(tr.test_type=='blank')].sort_values('t'); ba=relang(b.trial_id); z=np.exp(1j*ba)
        m=(st.session==sess).values; t=tr.set_index('trial_id').loc[st.trial_id[m],'hrs'].values
        if mode=='interp':   # straight-line between neighbouring blanks
            base=np.angle(np.stack([np.interp(t,b.hrs,z[:,k].real)+1j*np.interp(t,b.hrs,z[:,k].imag) for k in range(z.shape[1])],1))
        elif mode=='trend':  # one smooth line per feature over the session (robust to a noisy single blank)
            deg=1 if len(b)>=3 else 0
            base=np.angle(np.stack([np.polyval(np.polyfit(b.hrs,z[:,k].real,deg),t)+1j*np.polyval(np.polyfit(b.hrs,z[:,k].imag,deg),t) for k in range(z.shape[1])],1))
        elif mode=='prev':
            idx=np.searchsorted(b.hrs.values,t,side='right')-1; idx=np.clip(idx,0,len(b)-1); base=ba[idx]
        base0=np.angle(np.exp(1j*ba).mean(0))     # keep the session's average blank as reference level
        out[m]=a[m]-base+base0
    return toX(out)
h=stamps(tr,['healthy_vertebra']); s=stamps(tr,['severe_scoliosis'])
xs,ys=locate.mold('severe'); o=np.argsort(ys); S=dict(zip([0,30,60,85,125],xs[o])); Y=ys[o]
yh=h.x_mm.values.astype(float); gh=h.y_mm.values; x=s.x_mm.values.astype(float); Sv=s.y_mm.map(S).values
xa,ya=locate.mold('aligned'); ya=np.sort(ya)
print('blanks per session',tr[tr.test_type=='blank'].groupby('session').size().to_dict())
for mode in ['none','session-mean','stamp-trend']:
    Xh=corrected(h,mode); Xs=corrected(s,mode)
    raw=locate.grouped_cv(Xh,yh,gh); sl,ic=np.polyfit(yh,raw,1)
    m=locate.ridge_fit(Xh,yh,locate.pick_alpha(Xh,yh,gh)); p=(locate.ridge_predict(m,Xs)-ic)/sl
    co,*_=np.linalg.lstsq(np.c_[x,Sv,np.ones_like(x)],p,rcond=None)
    sp=pd.Series(x-p).groupby(s.y_mm.values).median().values
    # centred passes C-bow
    sv=pd.DataFrame({'y':s.y_mm.values,'r':s.repeat.values,'v':(x-p)})[x==0].pivot_table(index='y',columns='r',values='v')
    hc=(raw-ic)/sl; hv=pd.DataFrame({'y':gh,'r':h.repeat.values,'v':yh-hc})[yh==0].pivot_table(index='y',columns='r',values='v').loc[[0,20,60,80,100]].dropna(axis=1)
    bs=[fit_c(Y,sv[c].values)[1] for c in sv]; bh=[fit_c(ya,hv[c].values)[1] for c in hv]
    print(f'{mode:7s} healthy CV MAE {np.mean(abs(raw-yh)):.2f} | severe MAE {np.mean(abs(p-(x-Sv))):.2f} pred={co[0]:.2f}x{co[1]:+.2f}S | spine {np.round(sp,1)} Cobb {fit_segments(Y,sp)[0]:.1f} | centred bow severe {np.round(bs,1)} healthy {np.round(bh,1)}')
