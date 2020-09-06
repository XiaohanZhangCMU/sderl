import pandas as pd
import numpy as np
from sderl.utils.plot import make_plots
from matplotlib import pyplot as plt
import seaborn as sns

condition = "Condition1"
xaxis = 'Epoch'
values = ['mse']
value = values[0]
data_loc = './data/wind/QuadraticGradients/04Sep20'
#data_loc = './data/wind/ReactionDiffusion/04Sep20'
#data_loc = './data/wind/HJB/04Sep20'
#data_loc = './data/wind/BurgesType/04Sep20'


data, _, _ = make_plots([data_loc], xaxis=xaxis,  values=values, smooth=2, estimator='mean', shorten_legends=True, perf='loss', train_history_file='progress.csv', just_data = True)

print('Merge {0} progress.csv'.format(len(data)))
min_epochs = np.min([ d.shape[0] for d in data ])
data = pd.concat(data, ignore_index=True)
data['Epoch'] = data['Epoch'] * 20 # 20 = n_train_points // config.batch_size
min_epochs *= 20
sns.set(style="darkgrid", font_scale=1.5)


#ax = sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd')
ax = sns.lineplot(data=data, x=xaxis, y=value, ci='sd')

ax.set_yscale('log', nonposy='clip')
#plt.legend(loc='best').set_draggable(True)
ax.set_xlim([0, min_epochs])

# x-axis scale in scientific notation if max x is large
xscale = np.max(np.asarray(data[xaxis])) > 5e3
if xscale:
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.tight_layout(pad=0.5)
plt.show()

