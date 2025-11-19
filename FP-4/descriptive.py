from IPython.display import display,Markdown #,HTML
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd

from parse_data import parsedata

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )

df = parsedata()

df.describe()

def central(x, print_output=True):
    x0     = np.mean( x )
    x1     = np.median( x )
    x2     = stats.mode( x ).mode
    return x0, x1, x2


def dispersion(x, print_output=False):
    y0 = np.std( x ) # standard deviation
    y1 = np.min( x )  # minimum
    y2 = np.max( x )  # maximum
    y3 = y2 - y1      # range
    y4 = np.percentile( x, 25 ) # 25th percentile (i.e., lower quartile)
    y5 = np.percentile( x, 75 ) # 75th percentile (i.e., upper quartile)
    y6 = y5 - y4 # inter-quartile range
    
    if print_output:
        print("Standard Deviation   = ", y0)
        print("Minimum              = ", y1)
        print("Maximum              = ", y2)
        print("Range                = ", y3)
        print("25th percentile      = ", y4)
        print("75th percentile      = ", y5)
        print("Inter-quartile Range = ", y6)
    
    return y0, y1, y2, y3, y4, y5, y6

df_numerical = df.drop(['age group', 'blood defect type'], axis=1)

print(df_numerical.describe())

def display_central_tendency_table(num=1):
    display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
    df_central = df_numerical.apply(lambda x: central(x), axis=0)
    round_dict = {'sex': 2, 'age': 1, 'blood pressure': 2, 'cholesterol': 2, 'restecg':2, 'thalach': 2, 'exercise induced angina': 2, 'severity of artery blockage': 2}
    df_central = df_central.round( round_dict )
    row_labels = 'mean', 'median', 'mode'
    df_central.index = row_labels
    display( df_central )

display_central_tendency_table()

def display_dispersion_table(num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    round_dict = {'sex': 2, 'age': 1, 'blood pressure': 2, 'cholesterol': 2, 'restecg':2, 'thalach': 2, 'exercise induced angina': 2, 'severity of artery blockage': 2}
    df_dispersion         = df_numerical.apply(lambda x: dispersion(x), axis=0).round( round_dict )
    row_labels_dispersion = 'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    df_dispersion.index   = row_labels_dispersion
    display( df_dispersion )

display_dispersion_table(num=2)

bp    = df['blood pressure']
chol = df['cholesterol']
ang = df['exercise induced angina']
sev = df['severity of artery blockage']
ecg = df['restecg']
maxhr = df['thalach']
fbs = df['Fasting Blood Sugar']

fig,axs = plt.subplots( 1, 3, figsize=(10,3), tight_layout=True )
axs[0].scatter( bp, sev, alpha=0.5, color='b' )
axs[1].scatter( chol, sev, alpha=0.5, color='r' )
axs[2].scatter( maxhr, sev, alpha=0.5, color='g' )

xlabels = 'Blood Pressure', 'Cholesterol', 'Maximum Heart Rate' 
[ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
axs[0].set_ylabel('Stage of Disease')
[ax.set_yticklabels([])  for ax in axs[1:]]
plt.show()

def corrcoeff(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

def plot_regression_line(ax, x, y, **kwargs):
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)

fig,axs = plt.subplots( 1, 3, figsize=(10,3), tight_layout=True )
ivs     = [bp, chol, maxhr]
colors  = 'b', 'r', 'g'
for ax,x,c in zip(axs, ivs, colors):
    ax.scatter( x, sev, alpha=0.5, color=c )
    plot_regression_line(ax, x, sev, color='k', ls='-', lw=2)
    r   = corrcoeff(x, sev)
    ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

xlabels = 'Blood Pressure', 'Cholesterol', 'Maximum Heart Rate' 
[ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
axs[0].set_ylabel('Stage of Disease')
[ax.set_yticklabels([])  for ax in axs[1:]]
plt.show()

i_high   = sev > -2
i_low    = sev <= -2

fig,axs = plt.subplots( 1, 2, figsize=(8,3), tight_layout=True )
i = [chol]
for ax,i in zip(axs, [i_high, i_low]):
    ax.scatter( chol[i], sev[i], alpha=0.5, color='g' )
    plot_regression_line(ax, chol[i], sev[i], color='k', ls='-', lw=2)
[ax.set_xlabel('Cholesterol')  for ax in axs] 
axs[0].set_title('Severe Disease')
axs[0].set_ylabel('Severity of Disease')
axs[1].set_title('Not Severe Disease')
axs[1].set_xticks([200, 300, 400, 500])
plt.show()

def plot_cholesterol():
    
    fig,axs = plt.subplots( 1, 2, figsize=(8,3), tight_layout=True )
    for ax,i in zip(axs, [i_high, i_low]):
        ax.scatter( chol[i], sev[i], alpha=0.5, color='g' )
        plot_regression_line(ax, chol[i], sev[i], color='k', ls='-', lw=2)
    [axs[0].plot(chol[i_high].mean(), q, 'ro')  for q in [0]]
    [axs[1].plot(chol[i_low].mean(), q, 'ro')  for q in [-1, -2, -3]]
    [ax.set_xlabel('Cholesterol')  for ax in axs] 
    axs[0].set_title('High Severity')
    axs[0].set_ylabel('Severity of Disease')
    axs[1].set_title('Low Severity')
    axs[1].set_xticks([200, 300, 400, 500])
    plt.show()

def plot_descriptive():
    
    fig,axs = plt.subplots( 2, 2, figsize=(8,6), tight_layout=True )
    ivs     = [bp, chol, maxhr]
    colors  = 'b', 'r', 'g'
    for ax,x,c in zip(axs.ravel(), ivs, colors):
        ax.scatter( x, sev, alpha=0.5, color=c )
        plot_regression_line(ax, x, sev, color='k', ls='-', lw=2)
        r   = corrcoeff(x, sev)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    xlabels = 'Blood Pressure', 'Cholesterol', 'Maximum Heart Rate' 
    [ax.set_xlabel(s) for ax,s in zip(axs.ravel(),xlabels)]
    [ax.set_ylabel('Severity of Disease') for ax in axs[:,0]]
    [ax.set_yticklabels([])  for ax in axs[:,1]]


    ax       = axs[1,1]
    i_high    = sev > -1
    i_low   = sev <= -1
    fcolors  = 'm', 'c'
    labels   = 'High-Severity', 'Low-Severity'
    q_groups = [[0], [-1, -2, -3]]
    ylocs    = 0.3, 0.7
    for i,c,s,qs,yloc in zip([i_low, i_high], fcolors, labels, q_groups, ylocs):
        ax.scatter( chol[i], sev[i], alpha=0.5, color=c, facecolor=c, label=s )
        plot_regression_line(ax, chol[i], sev[i], color=c, ls='-', lw=2)
        [ax.plot(chol[i].mean(), q, 'o', color=c, mfc='w', ms=10)  for q in qs]
        r   = corrcoeff(chol[i], sev[i])
        ax.text(0.7, yloc, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    ax.legend()
    ax.set_xlabel('Cholesterol')

    panel_labels = 'a', 'b', 'c', 'd'
    [ax.text(0.02, 0.92, f'({s})', size=12, transform=ax.transAxes)  for ax,s in zip(axs.ravel(), panel_labels)]
    plt.show()
    
    display_title('Correlations amongst main variables.', pref='Figure', num=1)

    
plot_descriptive()

def plot_agedistribution():
    x        = df['age']
    x0,x1,x2 = central(x)
    
    custom_bins = [ 30, 40, 50, 60, 70, 80 ]
    
    plt.figure()
    plt.hist( x, bins=custom_bins)
    plt.axvline(x0, color='r', label='Mean')
    plt.axvline(x1, color='g', label='Median')
    plt.axvline(x2, color='k', label='Mode')
    axs[1].set_xticks([30, 40, 50, 60, 70])
    plt.legend()
    plt.show()

    display_title('Distribution of Age', pref='Figure', num=2)

plot_agedistribution()


n_female = (df['sex']==0).sum()
n_male    = (df['sex']==1).sum()

print('Female Patients', n_female)
print('Male Patients', n_male)

fig,axs = plt.subplots( 1, 2, figsize=(10,3), tight_layout=True )
axs[0].scatter( chol, bp, alpha=0.5, color='b' )
axs[1].scatter( fbs, bp, alpha=0.5, color='r' )


xlabels = 'Cholesterol', 'Fasting Blood Sugar Levels'
[ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
axs[0].set_ylabel('Blood Pressure')
[ax.set_yticklabels([])  for ax in axs[1:]]
plt.show()

x        = df['blood pressure']
x0,x1,x2 = central(x)

plt.figure()
plt.hist( x )
plt.axvline(x0, color='r', label='Mean')
plt.axvline(x1, color='g', label='Median')
plt.axvline(x2, color='k', label='Mode')
plt.legend()
plt.show()

age0   = df['age'][df['sex']==0]
age1   = df['age'][df['sex']==1]

print( f'Average age (st.dev) of female patients: {age0.mean():.1f} ({age0.std():.1f})'  )
print( f'Average age (st.dev) of male patients:    {age1.mean():.1f} ({age1.std():.1f})'  )

def plot_ecg():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sex_labels = {0: 'Female', 1: 'Male'}

    counts1 = df[df['restecg'] == 1]['sex'].value_counts().sort_index()
    axes[0].bar([sex_labels[i] for i in counts1.index], counts1.values)
    axes[0].set_xlabel('Sex')
    axes[0].set_ylabel('Count')
    axes[0].set_title('ST-T Wave Abnormality')
    axes[0].set_yticks([20, 40, 60, 80, 100])

    counts2 = df[df['restecg'] == 2]['sex'].value_counts().sort_index()
    axes[1].bar([sex_labels[i] for i in counts2.index], counts2.values)
    axes[1].set_xlabel('Sex')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Left Ventricular Hypertrophy')

    plt.tight_layout()
    plt.show()

    display_title('Heart Activity Abnormality in Female vs Male Patients', pref='Figure', num=3)
    
plot_ecg()
