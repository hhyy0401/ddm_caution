"""Conceptual task/DDM diagram. All trajectories are illustrative, not data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / 'figs'
plt.rcParams.update({'font.family':'serif', 'font.size':8,
                     'pdf.fonttype':42, 'svg.fonttype':'none'})
ink, blue, orange = '#253442', '#276C9B', '#B6632F'
fig = plt.figure(figsize=(3.38, 3.12), facecolor='white')
ax = fig.add_axes([.02,.48,.96,.51]); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(0,.98,'A  Suboptimality task',ha='left',va='top',weight='normal',color='black',fontsize=8)
ax.text(0,.79,'Error loss (points)',va='center',fontsize=8,color=ink)
for x,label in zip(np.linspace(.46,.95,5),['5','10','20','40','80']):
    ax.text(x,.79,label,ha='center',va='center',fontsize=8,color=orange,
            bbox=dict(boxstyle='round,pad=.2',fc='#FBF1E9',ec='none'))
ax.text(0,.63,'Correct response: +20 points',color=blue,fontsize=8)
def box(x,w,label):
    ax.add_patch(FancyBboxPatch((x,.31),w,.21,boxstyle='round,pad=.01',
                              facecolor='#F1F5F8',edgecolor='#A3B4C1',linewidth=.6))
    ax.text(x+w/2,.415,label,ha='center',va='center',fontsize=8,color=ink)
box(.02,.28,'Choice task');box(.38,.25,'Response');box(.71,.27,'Feedback')
for x,y in [(.31,.37),(.64,.70)]:
    ax.annotate('',xy=(y,.415),xytext=(x,.415),arrowprops=dict(arrowstyle='->',color=ink,lw=.8))
ax.annotate('',xy=(.16,.30),xytext=(.85,.30),arrowprops=dict(arrowstyle='->',color=ink,lw=.8,connectionstyle='bar,fraction=-.10'))
ax.text(.50,.075,'Repeat within a 60-s block',ha='center',fontsize=8,color=ink)
ax.text(.5,-.045,'No load  |  Concurrent letter memory',ha='center',fontsize=8,color=blue)

ax2=fig.add_axes([.16,.12,.69,.27]); ax2.set_xlim(0,1.03);ax2.set_ylim(-.10,1.1)
ax2.spines[['top','right','left','bottom']].set_visible(False);ax2.spines['bottom'].set_color('#8496A4')
ax2.set_xticks([]);ax2.set_yticks([])
ax2.axhline(0,color=ink,lw=.8);ax2.axhline(1,color=ink,lw=.8)
ax2.text(.99,1.025,'Correct',ha='right',va='bottom',fontsize=7,color=ink)
ax2.text(.99,-.025,'Error',ha='right',va='top',fontsize=7,color=ink)
tx=np.array([0,.04,.09,.15,.21,.27,.31,.36,.41,.46,.52,.59,.64,.69,.75,.79])
yy=np.array([.5,.54,.49,.58,.57,.64,.59,.70,.73,.69,.79,.83,.78,.91,.94,1.0])
ax2.plot(tx,yy,color=blue,lw=1.1)
ax2.plot([0,.79],[.5,1],ls='--',color=orange,lw=1)
ax2.text(.65,.27,r'$v$: drift rate',fontsize=8,color=orange,ha='center',va='center')
ax2.annotate('',xy=(-.055,1),xytext=(-.055,0),xycoords='data', annotation_clip=False,
             arrowprops=dict(arrowstyle='<->',lw=.8,color=ink))
ax2.text(-.16,.5,r'$a$',ha='center',va='center',fontsize=10,transform=ax2.transAxes)
ax2.set_xlabel('Decision time',labelpad=9,fontsize=8,color=ink)
fig.text(.02,.41,'B  Caution and evidence accumulation',weight='normal',fontsize=8,color='black')
fig.text(.02,.015,'Schematic only; not stimulus screenshots or fitted data.',fontsize=6.5,color='#566671')
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / 'fig0_task.pdf')
SOURCE = Path(__file__).resolve().parent / 'figure_sources'
SOURCE.mkdir(exist_ok=True)
for ext in ('svg','png'):
    fig.savefig(SOURCE / f'fig0_task.{ext}',dpi=220)
