import pandas as pd
import requests, time, openreview

url = 'https://api2.openreview.net/notes?content.venueid=TMLR'
df = pd.DataFrame(requests.get(url).json()['notes'])
df.to_csv('tmlr_notes.csv')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

submissions = sorted(list(df['cdate']))
ms_in_day = 1000 * 3600 * 24
interarrival_times = [  (submissions[i+1] - submissions[i])/ms_in_day for i in range(len(df)-1)]
mean_arrival_time = np.mean(interarrival_times)
plt.hist(interarrival_times, bins=100)
plt.yscale('log')
plt.xlabel("Inter-arrival times (days)")
plt.ylabel("Frequency")

plt.gca().set_facecolor('lightgray')
plt.grid(True, color='white', linestyle='-', linewidth=0.5)
plt.axvline(mean_arrival_time, color='black', linestyle='--', label=f'Mean interarrival time: {mean_arrival_time:.2f} days')
plt.legend(loc='upper right')
#plt.show()

import random
from scipy import stats

random.shuffle(interarrival_times)
split_idx = len(interarrival_times)//5
mean_sample, ks_sample = interarrival_times[:split_idx], interarrival_times[split_idx:]
_, ks_p_value = stats.kstest(ks_sample,
                             lambda x:stats.expon.cdf(x, scale=np.mean(mean_sample)))
print(ks_p_value, np.mean(mean_sample), np.mean(ks_sample))


