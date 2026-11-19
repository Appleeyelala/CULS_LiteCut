"""Plot all measured single edits, region-only blocks and the submitted joint choices."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(tables,output):
    singles=rows(tables/'isolated.csv')
    owners=rows(tables/'owners.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.15),layout='constrained')
    for ax,name in zip(axes,('c5315','intra_pred_PE')):
        case=[r for r in singles if r['circuit']==name]
        groups=[r for r in owners if r['circuit']==name and r['method'].startswith('only_')]
        control=next(r for r in case if r['method']=='original')
        joint=next(r for r in case if r['method']=='joint')
        one=[r for r in case if r['method'].startswith('single_')]
        for x,items,color in ((0,one,'#226a89'),(1,groups,'#42927d')):
            items=sorted(items,key=lambda r:(float(r['raw_ps']),r['method']))
            # Position equal observations side by side so ties remain countable.
            by_value={}
            for r in items:
                by_value.setdefault(float(r['raw_ps']),[]).append(r)
            for y,ties in by_value.items():
                centers=[x + (j-(len(ties)-1)/2)*min(.042,.42/max(1,len(ties)-1)) for j in range(len(ties))]
                ax.scatter(centers,[y]*len(ties),s=16,color=color,edgecolors='white',linewidths=.3,zorder=4)
        ax.scatter([2],[float(joint['raw_ps'])],marker='D',s=38,color='#a83c3b',zorder=5)
        ax.axhline(float(control['original_ps']),color='#555555',linestyle='--',linewidth=.85)
        ax.axhline(float(control['noop_ps']),color='#9b762c',linestyle=':',linewidth=1.1)
        ax.set_title(name,fontsize=10)
        ax.set_xticks([0,1,2], [f'One edit\n(n={len(one)})',f'One MFFC\n(n={len(groups)})','Joint\n(16 edits)'])
        ax.set_xlim(-.5,2.45)
        ax.tick_params(labelsize=8)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('ABC critical delay (ps)',fontsize=9)
    axes[1].legend(handles=[Line2D([],[],color='#555555',ls='--',lw=.85,label='Direct original'),
                            Line2D([],[],color='#9b762c',ls=':',lw=1.1,label='Zero-edit replay')],
                    frameon=False,fontsize=7,loc='best')
    fig.savefig(output,format='pdf',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
