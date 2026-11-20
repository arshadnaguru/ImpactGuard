"""Redraw the README figures in figures/ from results/main_split.md, results/event_split.md and results/logs/.

    python scripts/make_readme_figures.py
"""
import re, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=sys.argv[1] if len(sys.argv) > 1 else "."   # repo root
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
OURS="#B45309"; NAVY="#1F2F55"; GREY="#9AA3B2"; MID="#5B6C99"
def parse(p):
    rows={}
    for l in open(p):
        if l.startswith("| ") and "Method" not in l:
            c=[x.strip() for x in l.strip().strip("|").split("|")]
            if len(c) < 9: continue
            f=lambda s:(float(s.split("±")[0]),float(s.split("±")[1]) if "±" in s else 0.0) if s!="-" else (np.nan,0)
            rows[c[0]]=dict(auroc=f(c[2]),f1=f(c[3]),acc=f(c[4]),a10=f(c[5]),a30=f(c[6]),a60=f(c[7]),mae=f(c[8]))
    return rows
M=parse(f"{R}/results/main_split.md"); E=parse(f"{R}/results/event_split.md")
order=["Majority class","Mean-pool MLP","Static GCN (Ckpt 2)","Static GCN + prefix training","GIN (2019)","GAT (2018)","ImpactGuard-T (ours)","Bi-GCN (2020)","Bi-GCN + prefix training","TF-IDF + LogReg"]
# 1) main results: AUROC on both splits
fig,ax=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
for a,T,title in [(ax[0],M,"Main split (871 test cascades)"),(ax[1],E,"Unseen event: Ferguson held out (1,143)")]:
    y=np.arange(len(order)); v=[T[k]["auroc"][0] for k in order]; s=[T[k]["auroc"][1] for k in order]
    col=[OURS if "ours" in k else (MID if "Ckpt 2" in k else GREY) for k in order]
    a.barh(y,v,xerr=s,color=col,height=0.65,error_kw=dict(lw=1,capsize=2))
    for yi,vi,si in zip(y,v,s): a.text(vi+si+0.006,yi,f"{vi:.3f}",va="center",fontsize=9)
    a.set_yticks(y,order); a.set_xlim(0.45,0.95); a.set_title(title,fontsize=11); a.set_xlabel("Test AUROC (full cascade)")
plt.suptitle("Rumour detection on PHEME-RnR, mean ± std over 3 seeds",fontsize=12)
plt.tight_layout(); plt.savefig(f"{R}/figures/results_auroc.png",dpi=160); plt.close()
# 2) early detection
fig,ax=plt.subplots(1,2,figsize=(11,4.1),gridspec_kw={"width_ratios":[1.45,1]})
a=ax[0]; xs=["10 min","30 min","60 min","full"]
for k,c,ls in [("TF-IDF + LogReg","#6A7179",":"),("Bi-GCN (2020)",NAVY,"--"),("GAT (2018)",MID,"-."),("Static GCN (Ckpt 2)",GREY,"-"),("ImpactGuard-T (ours)",OURS,"-")]:
    vals=[M[k]["a10"][0],M[k]["a30"][0],M[k]["a60"][0],M[k]["auroc"][0]]
    a.plot(xs,vals,marker="o",color=c,ls=ls,lw=2.4 if "ours" in k else 1.6,label=k)
a.set_ylabel("Test AUROC"); a.set_title("How early can each model tell? (main split)",fontsize=11); a.set_ylim(0.82,0.91); a.legend(fontsize=8.5,loc="lower right",frameon=False)
a=ax[1]; names=["Train mean","Linear fit on\nobserved size","ImpactGuard-T\n(ours)"]; v=[M["Majority class"]["mae"][0],M["TF-IDF + LogReg"]["mae"][0],M["ImpactGuard-T (ours)"]["mae"][0]]
a.bar(names,v,yerr=[0,0,M["ImpactGuard-T (ours)"]["mae"][1]],color=[GREY,MID,OURS],width=0.6,capsize=3)
for i,vi in enumerate(v): a.text(i,vi+0.015,f"{vi:.3f}",ha="center",fontsize=9)
a.set_ylabel("MAE of log(1 + final size), lower is better"); a.set_title("Impact: final size predicted at 30 min",fontsize=11); a.set_ylim(0,0.8)
plt.tight_layout(); plt.savefig(f"{R}/figures/early_detection_and_impact.png",dpi=160); plt.close()
# 3) training curves (seed 42 logs)
fig,ax=plt.subplots(1,2,figsize=(11,3.6))
for name,lab,c in [("impactguard_seed42","ImpactGuard-T (ours)",OURS),("gcn_seed42","Static GCN (Ckpt 2)",GREY),("bigcn_seed42","Bi-GCN (2020)",NAVY)]:
    L=[l for l in open(f"{R}/results/logs/{name}.log") if l.startswith("epoch ")]
    ep=[int(re.search(r"epoch (\d+)",l).group(1)) for l in L]
    lo=[float(re.search(r"train loss ([\d.]+)",l).group(1)) for l in L]
    va=[float(re.search(r"val AUROC full ([\d.]+)",l).group(1)) for l in L]
    best=[e for e,l in zip(ep,L) if "saved best" in l][-1]
    ax[0].plot(ep,lo,color=c,label=lab,lw=2 if "ours" in lab else 1.4); ax[1].plot(ep,va,color=c,label=lab,lw=2 if "ours" in lab else 1.4)
    ax[1].scatter([best],[va[ep.index(best)]],color=c,zorder=3,s=28)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_title("Training loss (seed 42)",fontsize=11)
ax[0].text(0.98,0.95,"ours includes the size loss,\nso its scale is higher",transform=ax[0].transAxes,ha="right",va="top",fontsize=8,color="#6A7179")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation AUROC"); ax[1].set_title("Validation AUROC (dot = saved best.pt)",fontsize=11); ax[1].legend(fontsize=8.5,frameon=False,loc="lower right")
plt.tight_layout(); plt.savefig(f"{R}/figures/training_curves.png",dpi=160); plt.close()
# 4) confusion matrices on the test set (seed 42), ours vs the Ckpt 2 baseline
import ast
fig,ax=plt.subplots(1,2,figsize=(8.4,3.6))
for a,(name,lab) in zip(ax,[("gcn_seed42","Static GCN (Ckpt 2)"),("impactguard_seed42","ImpactGuard-T (ours)")]):
    line=[l for l in open(f"{R}/results/logs/{name}.log") if l.startswith("Confusion matrix")][0]
    cm=np.array(ast.literal_eval(line[line.index("[["):].strip()))
    a.imshow(cm,cmap="Oranges" if "ours" in lab else "Blues")
    for i in range(2):
        for j in range(2):
            a.text(j,i,f"{cm[i,j]}\n({cm[i,j]/cm[i].sum():.0%})",ha="center",va="center",fontsize=11,color="white" if cm[i,j]>cm.max()*0.55 else "black")
    a.set_xticks([0,1],["non-rumour","rumour"]); a.set_yticks([0,1],["non-rumour","rumour"])
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(lab,fontsize=11)
    for sp in a.spines.values(): sp.set_visible(False)
plt.suptitle("Test confusion matrices, seed 42 (871 threads)",fontsize=11.5)
plt.tight_layout(); plt.savefig(f"{R}/figures/confusion_matrices.png",dpi=160); plt.close()
print("ok")
