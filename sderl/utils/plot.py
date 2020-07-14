import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def mean_std_plot(log_dirs=[], is_smooth=False, title='Learning Curve'):
    """
    Another version of plotting learning curves migrated from rl_baselines
    """
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn
    from matplotlib.ticker import FuncFormatter
    from stable_baselines.results_plotter import load_results, ts2xy # Need to adapt to sderl

    def millions(x, pos):
        """
        Formatter for matplotlib
        The two args are the value and tick position

        :param x: (float)
        :param pos: (int) tick position (not used here
        :return: (str)
        """
        return '{:.1f}M'.format(x * 1e-6)


    def moving_average(values, window):
        """
        Smooth values by doing a moving average

        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')


    def smooth(xy, window=50):
        x, y = xy
        if y.shape[0] < window:
            return x, y

        original_y = y.copy()
        y = moving_average(y, window)

        if len(y) == 0:
            return x, original_y

        # Truncate x
        x = x[len(x) - len(y):]
        return x, y

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--log-dirs', help='Log folder(s)', nargs='+', required=True, type=str)
    # parser.add_argument('--title', help='Plot title', default='Learning Curve', type=str)
    # parser.add_argument('--smooth', action='store_true', default=False,
    #                     help='Smooth Learning Curve')
    # args = parser.parse_args()

    results = []
    algos = []


    for folder in log_dirs:
        print('folder = ', folder)
        timesteps = load_results(folder) # Need to adapt to sderl
        results.append(timesteps)
        if folder.endswith('/'):
            folder = folder[:-1]
        algos.append(folder.split('/')[-1])

    min_timesteps = np.inf

    # 'walltime_hrs', 'episodes'
    for plot_type in ['timesteps']:
        xy_list = []
        for result in results:
            x, y = ts2xy(result, plot_type) # Need to adapt to sderl
            if is_smooth:
                x, y = smooth((x, y), window=100)
            n_timesteps = x[-1]
            if n_timesteps < min_timesteps:
                min_timesteps = n_timesteps
            # xy_list.append((x, y))
            xy_list.append(pd.DataFrame({'timesteps':x,'reward':y}))

        fig = plt.figure(title)
        # for i, (x, y) in enumerate(xy_list):
        #     print(algos[i])
        #     plt.plot(x[:min_timesteps], y[:min_timesteps], label=algos[i], linewidth=2)

        data = pd.concat(xy_list, ignore_index=True)
        data = data.sort_values(by='timesteps')
        data.set_index('timesteps', inplace=True)
        time_series_df = data # .reset_index(drop=True)
        smooth_path = time_series_df.rolling(5).mean()
        path_deviation = time_series_df.rolling(5).std()
        plt.plot(smooth_path, linewidth=2)
        plt.fill_between(path_deviation.index, (smooth_path-2*path_deviation)['reward'], (smooth_path+2*path_deviation)['reward'], color='b', alpha=.1)

        plt.title(title)
        plt.legend()
        if plot_type == 'timesteps':
            if min_timesteps > 1e6:
                formatter = FuncFormatter(millions)
                plt.xlabel('Number of Timesteps')
                fig.axes[0].xaxis.set_major_formatter(formatter)
    plt.show()



def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, shorten_legends=False, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)


    def commonprefix(m):
        "Given a list of pathnames, returns the longest common leading component"
        if not m: return ''
        s1 = min(m)
        s2 = max(m)
        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i]
        return s1

    # To change hue name in plot, rename the column name here.

    data['Condition1_short']=data['Condition1'].apply(lambda x : x[len(commonprefix(data['Condition1'].unique().tolist())):])
    data['Condition2_short']=data['Condition2'].apply(lambda x : x[len(commonprefix(data['Condition2'].unique().tolist())):])

    sns.set(style="darkgrid", font_scale=1.5)
    if shorten_legends:        
        # sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition+'_short', ci='sd', **kwargs)
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition+'_short', ci='sd', **kwargs)
    else:
        # sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)        

    # plt.legend(loc='best').set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})
    plt.legend(loc='lower right', ncol=1, handlelength=1,
               mode=None, borderaxespad=0., prop={'size': 12})

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    sderl.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', shorten_legends=False):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator, shorten_legends=shorten_legends)
    plt.show()

    return data, values, estimator


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args:
        logdir (strings): As many log directories (or prefixes to log
            directories, which the plotter will autocomplete internally) as
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one
            match for a given logdir prefix, and you will need to provide a
            legend string for each one of those matches---unless you have
            removed some of them as candidates via selection or exclusion
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis.
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the
            off-policy algorithms. The plotter will automatically figure out
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``,
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show
            curves from logdirs that do not contain these substrings.

    """

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est)

if __name__ == "__main__":
    main()
