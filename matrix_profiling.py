import matrixprofile as mp
import numpy as np
from matplotlib import pyplot as plt

from constants import DataDir
from util import get_mp_dataframe_from_file, get_host_connection_frequency

df = get_mp_dataframe_from_file(DataDir.all_tables)

# get January values
january_day = '2015-01-21'
jan_df = df[(df['stime'] < '2015-01-24') & (df['stime'] > '2015-01-20')]
hosts_dos_attacked = jan_df[jan_df['attack_cat'] == 'DoS']['dstip'].unique()
connection_frequencies_january = get_host_connection_frequency(jan_df, hosts_dos_attacked[0])

profile_by_minutes = 20
profile = mp.compute(connection_frequencies_january['connection_frequency'].values, profile_by_minutes, n_jobs=-1)
profile = mp.discover.discords(profile)

# We have to adjust the matrix profile to match the dimensions of the original
# time series
mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)

# Create a plot with three subplots
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20,7))
axes[0].plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
axes[0].set_title(f'Connections to {hosts_dos_attacked[0]} per minute on {january_day}', size=22)

#Plot the Matrix Profile
axes[1].plot(np.arange(len(mp_adjusted)), mp_adjusted)
axes[1].set_title('Matrix Profile', size=22)

for discord in profile['discords']:
    x = discord
    y = profile['mp'][discord]
    num_dos_attacks = connection_frequencies_january['dos_sum'][discord]

    axes[1].text(x, y, f'{num_dos_attacks}')
    axes[1].plot(x, y, marker='*', markersize=10, c='r')

plt.show()

# get February values
february_day = '2015-02-18'
feb_df = df[(df['stime'] < '2015-02-20') & (df['stime'] > '2015-02-17')]
hosts_dos_attacked = feb_df[feb_df['attack_cat'] == 'DoS']['dstip'].unique()
connection_frequencies_february = get_host_connection_frequency(feb_df, hosts_dos_attacked[0])

profile_by_minutes = 20
profile = mp.compute(connection_frequencies_february['connection_frequency'].values, profile_by_minutes, n_jobs=-1)
profile = mp.discover.discords(profile)

# We have to adjust the matrix profile to match the dimensions of the original
# time series
mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)

# Create a plot with three subplots
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20,7))
axes[0].plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
axes[0].set_title(f'Connections to {hosts_dos_attacked[0]} per minute on {february_day}', size=22)

#Plot the Matrix Profile
axes[1].plot(np.arange(len(mp_adjusted)), mp_adjusted)
axes[1].set_title('Matrix Profile', size=22)

for discord in profile['discords']:
    x = discord
    y = profile['mp'][discord]
    num_dos_attacks = connection_frequencies_february['dos_sum'][discord]

    axes[1].text(x, y, f'{num_dos_attacks}')
    axes[1].plot(x, y, marker='*', markersize=10, c='r')

plt.show()
