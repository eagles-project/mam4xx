import argparse
from glob import glob
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
from datetime import datetime
import subprocess 


parser = argparse.ArgumentParser()
parser.add_argument("--case1", type=str, help="Path to case1 run")
parser.add_argument("--casename1", type=str, help="Short name for case1 run")
parser.add_argument("--case2", type=str, help="Path to case2 run")
parser.add_argument("--casename2", type=str, help="Short name for case2 run")
parser.add_argument("--outdir", type=str, help="Output directory for images")
parser.add_argument("--html", type=str, help="Generate HTML file with images")

args = parser.parse_args()

case1 = args.case1
case2 = args.case2 
casename1 = args.casename1
casename2 = args.casename2 
outdir = args.outdir 
html = args.html 

def grab_timing(case):

    with open(f'{case}/log.job_stat', 'r') as f:
        lines = [line.strip() for line in f]
    jobid = lines[2].split()[0]
    status = lines[2].split()[5] 

    if status == 'COMPLETED':
        timinglog = glob(f'{case}/timing/e3sm_timing_stats.{jobid}.*')[0]      
        df = pd.read_csv(timinglog, sep='\s+', header=4)  
        s1 = df['name'].str.startswith('a:EAMxx::') 
        s2 = df['name'].str.endswith('::run')
        df2 = df[s1 & s2][1:][['name', 'walltotal']] 
        processes = [p for p in df2['name'].str.split('::').str[1]]  
        proc_new = [p not in ['EAMxx', 'physics', 'mac_aero_mic'] for p in processes]

        df3 = df2[proc_new].sort_values('name')
        processes = [p for p in df3['name'].str.split('::').str[1]] 
        
        compset = case.split('/')[-1].split('.')[2] 
        if compset == 'F2010-EAMxx-MAM4xx': aer_proc = [p.startswith('mam') for p in processes]
        if compset == 'F2010-SCREAMv1': aer_proc = [p.startswith('spa') for p in processes]

        dfaer = df3[aer_proc].reset_index(drop=True) 
        dfeam = df3[~np.array(aer_proc)] 
        dftmp = pd.DataFrame({'name': 'a:EAMxx::aerosols::run', 'walltotal': dfaer['walltotal'].sum()}, index=[99])
        dfeam = pd.concat([dfeam, dftmp]).reset_index(drop=True) 
        return dfeam, dfaer 
    else:
        print(f'Run {case} not completed')
        return None


def plot_mam4_process(dfeam, dfaer, axl, axr):

    eam_ratios = dfeam['walltotal'] / dfeam['walltotal'].sum() 
    labels = [p.replace('SurfaceCoupling', 'SC') for p in dfeam['name'].str.split('::').str[1]] 
    labels[labels.index('aerosols')] = 'mam4'
    
    explode = np.zeros(len(labels)) 
    explode[-1] = 0.08
    # rotate so that first wedge is split by the x-axis
    angle = eam_ratios.values[-1] / 2 * 360
    # angle = 0
    autopct = lambda v: f'{v:.1f}%' if v >= 1 else None
    colors = plt.get_cmap('Set3').colors[:]
    wedges, *_ = axl.pie(eam_ratios, startangle=angle, colors=colors, autopct=autopct, 
                        explode=explode)

    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"), zorder=0, va="center")
    l = []
    for i, p in enumerate(wedges):
        if eam_ratios[i] < 0.01: 
            l.append(f'{labels[i]} ({eam_ratios[i]:.1%})')  
            continue
        else: 
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle,"color":colors[i]})
            axl.annotate(labels[i], xy=(x, y), xytext=(1.2*np.sign(x), y),
                        horizontalalignment=horizontalalignment, **kw)
    axl.axis('equal') 
    # create handles and labels for legend, take only those where value is < 1
    handles = [h for h,r in zip(wedges,eam_ratios) if r < 0.01]
    axl.legend(handles, l, fontsize='small', ncols=2, bbox_to_anchor=(0.5,0.01), loc='lower center', 
            title='EAMxx process < 1%', title_fontsize='small', handlelength=0.5, handleheight=0.5, handletextpad=0.5, columnspacing=0.5)  
    axl.set_title('EAMxx/MAM4xx', fontsize='large', y=0.8)


    mam_ratios = dfaer['walltotal'] / dfaer['walltotal'].sum() 
    mam_labels = ['_'.join(p.split('_')[1:]) for p in dfaer['name'].str.split('::').str[1]]  
    wedges2, *_ = axr.pie(mam_ratios, autopct=autopct, colors=colors) 
    l2 = [] 
    for i, p in enumerate(wedges2):
        if mam_ratios[i] < 0.01: 
            l2.append(f'{mam_labels[i]} ({mam_ratios[i]:.1%})')
            continue
        else: 
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle,"color":colors[i]})
            axr.annotate(mam_labels[i], xy=(x, y), xytext=(1.15*np.sign(x), y),
                        horizontalalignment=horizontalalignment, **kw)
    handles = [h for h,r in zip(wedges2,mam_ratios) if r < 0.01]
    axr.legend(handles, l2, fontsize='small', ncols=2, bbox_to_anchor=(0.5,0.01), loc='lower center', 
            title='MAM4xx process < 1%', title_fontsize='small', handlelength=0.5, handleheight=0.5, handletextpad=0.5, columnspacing=0.5)  
    axr.set_title('MAM4xx', fontsize='large', y=0.8)
    axr.axis('equal')


def plot_eam_process(dfeam, ax):

    eam_ratios = dfeam['walltotal'] / dfeam['walltotal'].sum() 
    labels = [p.replace('SurfaceCoupling', 'SC') for p in dfeam['name'].str.split('::').str[1]] 
    labels[labels.index('aerosols')] = 'spa'
    
    explode = np.zeros(len(labels)) 
    explode[-1] = 0.08
    # rotate so that first wedge is split by the x-axis
    angle = eam_ratios.values[-1] / 2 * 360
    # angle = 0
    autopct = lambda v: f'{v:.1f}%' if v >= 1 else None
    colors = plt.get_cmap('Set3').colors[:]
    wedges, *_ = ax.pie(eam_ratios, startangle=angle, colors=colors, autopct=autopct, 
                        explode=explode)

    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"), zorder=0, va="center")
    l = []
    for i, p in enumerate(wedges):
        if eam_ratios[i] < 0.01: 
            l.append(f'{labels[i]} ({eam_ratios[i]:.1%})')  
            continue
        else: 
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle,"color":colors[i]})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.2*np.sign(x), y),
                        horizontalalignment=horizontalalignment, **kw)
            ax.axis('equal') 
            # create handles and labels for legend, take only those where value is < 1
    handles = [h for h,r in zip(wedges,eam_ratios) if r < 0.01]
    ax.legend(handles, l, fontsize='small', title='EAMxx process < 1%', title_fontsize='small', 
                handlelength=0.5, handleheight=0.5, handletextpad=0.5, columnspacing=0.5,
            #   ncols=2, bbox_to_anchor=(0.5,0.01), loc='lower center', 
            )  
    ax.set_title('EAMxx', fontsize='large')


t1, taer1 = grab_timing(case1)
t2, taer2 = grab_timing(case2) 

expname = '.'.join(os.path.basename(case1).split('.')[1:4]) 
now_str = datetime.now().strftime("%Y%m%d-%H%M%S")

if t1 is not None and t2 is not None:

    plt.rcParams.update({'font.size': 12})

    # plot eamxx process 
    fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 4))

    run_time = {}
    eamprocess = [p for p in t1['name'].str.split('::').str[1]]  
    run_time[casename1] = t1['walltotal']
    run_time[casename2] = t2['walltotal']     

    x = np.arange(len(eamprocess))
    width = 0.4
    multiplier = 0
    for cn, rt in run_time.items():
        offset = width * multiplier 
        rects = ax.bar(x + offset , rt, width, label=cn)

        multiplier += 1
    ax.legend(fontsize='large', loc='upper left', ) 
    eamproc = [p.replace('SurfaceCoupling', 'SC') for p in eamprocess] 
    ax.set_xticks(x + width / 2, eamproc)
    ax.set_xticklabels(eamproc, rotation=45, fontsize='large')
    ax.set_ylabel('wallclock', fontsize='large') 

    barplot = 'eamxx_process'
    image0 = f"{outdir}/{barplot}_{now_str}.png"
    fig.savefig(image0, dpi=300, bbox_inches='tight')
    plt.close(fig) 

    # plot mam4 process
    compset = case1.split('/')[-1].split('.')[2] 
    if compset == 'F2010-EAMxx-MAM4xx':
        fig, axs = plt.subplots(figsize=(10, 4.5), ncols=2, constrained_layout=True)
        plot_mam4_process(t1, taer1, axs[0], axs[1])
        piechart1 = 'process_case1'
        image1 = f"{outdir}/{piechart1}_{now_str}.png" 
        fig.savefig(image1, dpi=300, bbox_inches='tight') 
        plt.close(fig)

        fig, axs = plt.subplots(figsize=(10, 4.5), ncols=2, constrained_layout=True)
        plot_mam4_process(t2, taer2, axs[0], axs[1])
        piechart2 = 'process_case2'
        image2 = f"{outdir}/{piechart2}_{now_str}.png" 
        fig.savefig(image2, dpi=300, bbox_inches='tight') 
        plt.close(fig) 
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        plot_eam_process(t1, ax)
        piechart1 = 'process_case1'
        image1 = f"{outdir}/{piechart1}_{now_str}.png" 
        fig.savefig(image1, dpi=300, bbox_inches='tight') 
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        plot_eam_process(t2, ax)
        piechart2 = 'process_case2'
        image2 = f"{outdir}/{piechart2}_{now_str}.png" 
        fig.savefig(image2, dpi=300, bbox_inches='tight') 
        plt.close(fig) 


# Save the figure as an image file
html_content = "<html><head><title>Compare Model Performance</title></head><body>"

# Add the image to the HTML content
png_files = [image0, image1, image2] 
png_names = [expname, 'RUN1: '+casename1, 'RUN2: '+casename2]
for v, img in zip(png_names, png_files):
    img = f'{html}/{os.path.basename(img)}'
    html_content += f'<h2>{v}</h2>\n'
    html_content += f'<img src="{img}" style="max-width: 1000px;"><br><br>\n'
    html_content += "</body></html>"

# Write the HTML content to a file
html_file_path = outdir  
with open(f'{html_file_path}/compare_{casename1}_vs_{casename2}_{now_str}.html', "w") as html_file:
    html_file.write(html_content)

# Display the HTML file path
subprocess.run(['ln', '-sf', f'{html_file_path}/compare_{casename1}_vs_{casename2}_{now_str}.html', f'{html_file_path}/plot.html'], check=True)
print(f"HTML file generated: {html}/plot.html") 



 