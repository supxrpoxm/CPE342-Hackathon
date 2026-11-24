"""Extra Task 3 candidates (small L1 LGBM, CatBoost MAE, log-target L1) vs the notebook baseline.
Run from anywhere: python experiments/task3/exp_extra_models.py   (reuses cells 2,3,5 of task3.ipynb)"""
import json,os,sys,time
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]   # repo root (notebook cells expect it as cwd)
os.chdir(REPO)
nb=json.load(open('task3/task3.ipynb',encoding='utf-8'))
src=lambda i:''.join(nb['cells'][i]['source'])
g={}
for i in (2,3,5): exec(src(i),g)
exec("X_all = add_features(train.drop(columns=[ID, TARGET]))\ny = train[TARGET].values",g)
np=g['np'];lgb=g['lgb'];nmae=g['nmae'];X=g['X_all'];y=g['y'];KFold=g['KFold']
from catboost import CatBoostRegressor
kf=KFold(5,shuffle=True,random_state=42)
names=['l1_base','l1_small','cat_mae','log_l1']
oof={n:np.zeros(len(y)) for n in names}
for k,(tr,va) in enumerate(kf.split(X)):
    t=time.time()
    Xtr,Xva,ytr,yva=X.iloc[tr],X.iloc[va],y[tr],y[va]
    m=g['make_l1'](2000).fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric='l1',callbacks=[lgb.early_stopping(100,verbose=False)])
    oof['l1_base'][va]=np.clip(m.predict(Xva),0,None)
    m=lgb.LGBMRegressor(objective='l1',n_estimators=3000,learning_rate=0.03,num_leaves=15,min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=5,random_state=1,verbose=-1)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric='l1',callbacks=[lgb.early_stopping(100,verbose=False)])
    oof['l1_small'][va]=np.clip(m.predict(Xva),0,None)
    Xc,Xcv=Xtr.copy(),Xva.copy()
    for c in Xc.columns:
        if str(Xc[c].dtype)=='category': Xc[c]=Xc[c].astype(float); Xcv[c]=Xcv[c].astype(float)
    cb=CatBoostRegressor(loss_function='MAE',iterations=1500,learning_rate=0.08,depth=8,verbose=0,random_seed=1,thread_count=-1)
    cb.fit(Xc,ytr,eval_set=(Xcv,yva),early_stopping_rounds=100)
    oof['cat_mae'][va]=np.clip(cb.predict(Xcv),0,None)
    m=lgb.LGBMRegressor(objective='l1',n_estimators=3000,learning_rate=0.03,num_leaves=63,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=2,verbose=-1)
    m.fit(Xtr,np.log1p(ytr),eval_set=[(Xva,np.log1p(yva))],eval_metric='l1',callbacks=[lgb.early_stopping(100,verbose=False)])
    oof['log_l1'][va]=np.clip(np.expm1(m.predict(Xva)),0,None)
    print('fold',k,{n:round(nmae(yva,oof[n][va]),4) for n in names},round(time.time()-t),'s',flush=True)
for n in names: print(n,round(nmae(y,oof[n]),4))
avg=np.mean([oof[n] for n in ['l1_base','l1_small','cat_mae']],axis=0)
print('avg3',round(nmae(y,avg),4))
avg2=np.mean([oof[n] for n in ['l1_base','cat_mae']],axis=0)
print('avg l1_base+cat',round(nmae(y,avg2),4))
os.makedirs('task3/outputs',exist_ok=True)
np.save('task3/outputs/oof_extra_models.npy',np.vstack([oof[n] for n in names]))
