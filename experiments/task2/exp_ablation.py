"""Ablations on friend's task2 pipeline (LightGBM single model, 5-fold). Reuses friend's FE code verbatim."""
import json, sys, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
nb = json.load(open('task2/task2_friend.ipynb', encoding='utf8'))
src = lambda i: ''.join(nb['cells'][i]['source'])
train = pd.read_csv('task2/train.csv'); test = pd.read_csv('task2/test.csv')
TARGET_COL='segment'; RANDOM_SEED=42
exec(src(6))   # clean_data
KEEP = sys.argv[1].split(',') if len(sys.argv)>1 and sys.argv[1] else []
# friend's FE, with optional columns removed from its noise-drop list
fe = src(12).split('X_fe, y_fe, X_test_fe')[0]
for c in KEEP: fe = fe.replace(f"'{c}',", "").replace(f"'{c}'", "'__none__'") if False else fe
exec(fe)
def run(keep_cols, early_stop_on_val=True, tag=''):
    tr, te = clean_data(train, test)
    code = fe
    for c in keep_cols:
        code = code.replace(f"'{c}', ", "").replace(f"'{c}'\n", "'__x__'\n")
    ns = dict(globals()); exec(code, ns)
    X, y, _ = ns['build_leakfree_features'](tr, te)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros((len(X),4))
    for f,(a,b) in enumerate(skf.split(X,y),1):
        m = lgb.LGBMClassifier(objective='multiclass', n_estimators=3500, learning_rate=0.03, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.5, reg_lambda=2.5,
              random_state=RANDOM_SEED+f, n_jobs=16, verbose=-1)
        if early_stop_on_val:
            m.set_params(n_estimators=3500)
            m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(80, verbose=False)])
        else:  # inner split for early stopping: no val-fold peeking
            rng = np.random.RandomState(f); inner = rng.rand(len(a))<0.1
            m.fit(X.iloc[a[~inner]], y[a[~inner]], eval_set=[(X.iloc[a[inner]], y[a[inner]])], callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[b] = m.predict_proba(X.iloc[b])
    s = f1_score(y, oof.argmax(1), average='macro')
    print(f'{tag:45s} feats={X.shape[1]:3d} OOF macroF1={s:.5f}', flush=True)
    return s
if __name__=='__main__':
    run([], True,  'A friend FE (drop list as-is), ES on val')
    run([], False, 'B same, ES on inner split (no val peek)')
    run(['player_type_tag'], False, 'C + player_type_tag')
    run(['skill_tier','skill_tier_ord'], False, 'D + skill_tier')
    run(['region','payment_method','language','account_status'], False, 'E + region/payment/lang/status')
    run(['player_type_tag','skill_tier','skill_tier_ord','region','payment_method','language','account_status',
         'account_age_days','preferred_game_mode','avg_match_length'], False, 'F + ALL dropped columns')
