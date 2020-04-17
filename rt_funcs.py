#!/bin/bash

def get_cases(source_df, country, column="Country/Region"):
    import pandas as pd
    
    out = source_df[source_df[column] == country].iloc[:,4:].T
    out.columns = ["{}_cases".format(country)]
    return out.reindex(pd.to_datetime(out.index)).iloc[:,0]


def prepare_cases(cases, winsize, cases_cumulative=False):
    
    if cases_cumulative:
        new_cases = cases.diff()
    else:
        new_cases = cases

    smoothed = new_cases.rolling(winsize,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    zeros = smoothed.index[smoothed.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed.index.get_loc(last_zero) + 1
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed



def get_posteriors(sr, winsize=7, 
                   min_periods=1, 
                   R_T_MAX = 12,
                   GAMMA=1/4):
    import numpy as np
    import pandas as pd
    from scipy import stats as sps

    # We create an array for every possible value of Rt.
    r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)
    
    # Compute the Poisson lambdas for each Rt.
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # Note: if you want to have a Uniform prior you can use the following line instead.
    # I chose the gamma distribution because of our prior knowledge of the likely value
    # of R_t.
    
    # prior0 = np.full(len(r_t_range), np.log(1/len(r_t_range)))
    prior0 = np.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)

    likelihoods = pd.DataFrame(
        # Short-hand way of concatenating the prior and likelihoods
        data = np.c_[prior0, sps.poisson.logpmf(sr[1:].values, lam)],
        index = r_t_range,
        columns = sr.index)

    # Perform a rolling sum of log likelihoods. This is the equivalent
    # of multiplying the original distributions. Exponentiate to move
    # out of log.
    posteriors = likelihoods.rolling(winsize,
                                     axis=1,
                                     min_periods=min_periods).sum()
    posteriors = np.exp(posteriors)

    # Normalize to 1.0
    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)
    
    return posteriors




def plot_posteriors(posteriors):

    ax = posteriors.plot(title=f'{COUNTRY} - Daily Posterior for $R_t$',
               legend=False, 
               lw=1,
               c='k',
               alpha=.3,
               xlim=(0.4,4))

    ax.set_xlabel('$R_t$');
    
    
    
# Note that this takes a while to execute - it's not the most efficient algorithm
def highest_density_interval(pmf, p=.95):
    import pandas as pd
    import numpy as np
    
    
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col]) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i+1:]):
            if (high_value-value > p) and (not best or j<best[1]-best[0]):
                best = (i, i+j+1)
                break
            
    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=['Low', 'High'])




def get_data_for_rt_plot(posteriors):
    import pandas as pd
    
    hdis = highest_density_interval(posteriors)

    most_likely = posteriors.idxmax().rename('ML')

    # Look into why you shift -1
    result = pd.concat([most_likely, hdis], axis=1)
    result = result.reindex(result.index.rename("date"))

    return result




def process_data(new_cases, winsize, cases_cumulative=False): 
    original, smoothed = prepare_cases(new_cases, 
                                       winsize=winsize, 
                                       cases_cumulative=cases_cumulative)
    
    posteriors = get_posteriors(smoothed.dropna(), winsize=winsize)
    
    group_result = get_data_for_rt_plot(posteriors)
    
    return {'raw':original,
            'smoothed': smoothed, 
            'rt': group_result}



def plot_rt(result, ax, COUNTRY):
    import numpy as np
    import pandas as pd
    
    from matplotlib import pyplot as plt
    from matplotlib.dates import date2num
    from matplotlib import dates as mdates
    from matplotlib import ticker
    from matplotlib.colors import ListedColormap
    from scipy.interpolate import interp1d

    ax.set_title(f"{COUNTRY}")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25, label=COUNTRY)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result['High'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    extended = pd.date_range(start=index[0],
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

#     ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)