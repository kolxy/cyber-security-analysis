import matrixprofile as mp
import numpy as np
from matplotlib import pyplot as plt

from constants import DataDir
from util import get_clean_dataframe_from_file, get_host_connection_frequency

df = get_clean_dataframe_from_file(DataDir.all_tables)
hosts = df.unique('dstip')
connection_frequencies = get_host_connection_frequency(df, hosts[0])
profile_by_minutes = 25
print(df['attack_cat'].head)
print(df.dtypes)
profile = mp.compute(connection_frequencies['connection_frequency'].values, profile_by_minutes, n_jobs=-1)
profile = mp.discover.discords(profile)

# We have to adjust the matrix profile to match the dimensions of the original
# time series
mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)

# Create a plot with three subplots
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
ax.plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
ax.set_title('Connection Frequencies', size=22)
ax.set_xlabel('Time elapsed (minutes)')
ax.set_ylabel('Number of connections')

for discord in profile['discords']:
    print(connection_frequencies['dos_sum'][discord])
    x = np.arange(discord, discord + profile['w'])
    y = profile['data']['ts'][discord:discord + profile['w']]

    ax.plot(x, y, c='r')

plt.show()
