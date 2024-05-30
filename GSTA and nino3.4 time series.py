
######
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


# open the nc data
GSTA_path = 'gistemp1200_GHCNv4_ERSSTv5.nc'
GSTA = xr.open_dataset(GSTA_path)
print(GSTA)


# GSTA(Global Mean Surface Temperature Anomly) include Nino regions
GSTA_included = GSTA['tempanomaly'].mean(dim=['lat', 'lon'])
GSTA_array_in = GSTA_included.values


# Define the Nino regions to exclude
exclude_regions = [{'lat_bounds': (-10, 0), 'lon_bounds': (-90, -80)},  
                   {'lat_bounds': (-5, 5), 'lon_bounds': (160, 180)},
                   {'lat_bounds': (-5, 5), 'lon_bounds': (-180, -90)}]


# Mask the regions
for region in exclude_regions:
    lat_mask = (GSTA.lat >= region['lat_bounds'][0]) & (GSTA.lat <= region['lat_bounds'][1])
    lon_mask = (GSTA.lon >= region['lon_bounds'][0]) & (GSTA.lon <= region['lon_bounds'][1])
    GSTA['tempanomaly'] = GSTA['tempanomaly'].where(~(lat_mask & lon_mask), other=np.nan)


# GSTA excluding the Nino regions
GSTA_excluded = GSTA['tempanomaly'].mean(dim=['lat', 'lon'], skipna=True)
GSTA_array_ex = GSTA_excluded.values


# dates
dates = pd.date_range(start="1880-01-01", periods=len(GSTA_array_in), freq='MS')


# create a time series dataFrame
GSTA_in_series = pd.DataFrame(GSTA_array_in, index=dates, columns=['GSTA in'])
GSTA_ex_series = pd.DataFrame(GSTA_array_ex, index=dates, columns=['GSTA ex'])


# read the Nino3.4 index
nino34 = pd.read_csv('3.4.txt', delim_whitespace=True, index_col='year')
nino34_series = nino34.stack()
nino34_series.index = pd.to_datetime(nino34_series.index.map('{0[0]}-{0[1]}'.format))


# filter the data for the specific period
nino34_mask_1950_1975 = nino34_series.index.year.isin(range(1950, 1975))
filtered_nino34_1950_1975 = nino34_series[nino34_mask_1950_1975]

GSTA_in_mask_1950_1975= GSTA_in_series.index.year.isin(range(1950, 1975))
filtered_GSTA_in_1950_1975 = GSTA_in_series[GSTA_in_mask_1950_1975]

nino34_mask_1975_2000 = nino34_series.index.year.isin(range(1975, 2000))
filtered_nino34_1975_2000  = nino34_series[nino34_mask_1975_2000 ]

GSTA_in_mask_1975_2000 = GSTA_in_series.index.year.isin(range(1975, 2000))
filtered_GSTA_in_1975_2000  = GSTA_in_series[GSTA_in_mask_1975_2000 ]

nino34_mask_2000_2025 = nino34_series.index.year.isin(range(2000, 2025))
filtered_nino34_2000_2025  = nino34_series[nino34_mask_2000_2025 ]

GSTA_in_mask_2000_2025 = GSTA_in_series.index.year.isin(range(2000, 2025))
filtered_GSTA_in_2000_2025  = GSTA_in_series[GSTA_in_mask_2000_2025 ]

GSTA_ex_mask_1950_1975= GSTA_ex_series.index.year.isin(range(1950, 1975))
filtered_GSTA_ex_1950_1975 = GSTA_ex_series[GSTA_ex_mask_1950_1975]

GSTA_ex_mask_1975_2000= GSTA_ex_series.index.year.isin(range(1975, 2000))
filtered_GSTA_ex_1975_2000 = GSTA_ex_series[GSTA_ex_mask_1975_2000]

GSTA_ex_mask_2000_2025= GSTA_ex_series.index.year.isin(range(2000, 2025))
filtered_GSTA_ex_2000_2025 = GSTA_ex_series[GSTA_ex_mask_2000_2025]


# plot the time series Nino3.4 and GSTA include Nino regions
fig1 = plt.figure(figsize=(12,10))

ax1 = fig1.add_subplot(3,1,1)
ax1.set_ylim(-3, 3)
ax1.set_yticks([ -3, -2, -1, 0, 1, 2, 3])
ax1.axhline(0, color='BLACK', linewidth=1)
ax1.axhline(0.5, color='GRAY', linewidth=1.5)
ax1.axhline(-0.5, color='GRAY', linewidth=1.5)
ax1.plot(filtered_nino34_1950_1975.index, filtered_nino34_1950_1975.values, color = 'blue', label = 'Nino3.4 anomaly')
ax1.set_ylabel('Nino3.4', c='blue')
ax4 = ax1.twinx()
ax4.set_ylim(-1, 1)
ax4.set_yticks([-1, -0.5, 0, 0.5, 1])
ax4.plot(filtered_GSTA_in_1950_1975.index, filtered_GSTA_in_1950_1975.values, color = 'r', label = 'GSTA')
ax4.set_ylabel('Temperature', c='r')

ax2 = fig1.add_subplot(3,1,2)
ax2.set_ylim(-3, 3)
ax2.set_yticks([ -3, -2, -1, 0, 1, 2, 3])
ax2.axhline(0, color='black', linewidth=1)
ax2.axhline(0.5, color='GRAY', linewidth=1.5)
ax2.axhline(-0.5, color='GRAY', linewidth=1.5)
ax2.plot(filtered_nino34_1975_2000.index, filtered_nino34_1975_2000.values, color = 'blue')
ax2.set_ylabel('Nino3.4', c='blue')
ax5 = ax2.twinx()
ax5.set_ylim(-1.5, 1.5)
ax5.set_yticks([-1.5,-0.75, 0, 0.75,1.5])
ax5.plot(filtered_GSTA_in_1975_2000.index, filtered_GSTA_in_1975_2000.values,  color = 'r')
ax5.set_ylabel('Temperature', c='r')

ax3 = fig1.add_subplot(3,1,3)
ax3.set_ylim(-3, 3)
ax3.set_yticks([ -3, -2, -1, 0, 1, 2, 3])
ax3.axhline(0, color='black', linewidth=1)
ax3.axhline(0.5, color='GRAY', linewidth=1.5)
ax3.axhline(-0.5, color='GRAY', linewidth=1.5)
ax3.plot(filtered_nino34_2000_2025.index, filtered_nino34_2000_2025.values, color = 'blue')
ax3.set_ylabel('Nino3.4', c='blue')
ax3.set_xlabel('Time(year)')
ax6 = ax3.twinx()
ax6.set_ylim(-2.5, 2.5)
ax6.set_yticks([-2.5,-1.5, -1, 0, 1.5,2.5])
ax6.plot(filtered_GSTA_in_2000_2025.index, filtered_GSTA_in_2000_2025.values, color = 'r')
ax6.set_ylabel('Temperature', c='r')

fig1.legend()
plt.suptitle('Monthly Nino3.4 and Global Mean Surface Temperature anomaly Time serise (1950-2024)')
plt.tight_layout() 
plt.savefig('Monthly Nino3.4 and Global Mean Surface Temperature anomaly Time serise.jpg', dpi=300, bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(10,5))

nino34_mask_2023_2025 = nino34_series.index.year.isin(range(2023, 2025))
filtered_nino34_2023_2025  = nino34_series[nino34_mask_2023_2025 ]

GSTA_in_mask_2023_2025 = GSTA_in_series.index.year.isin(range(2023, 2025))
filtered_GSTA_in_2023_2025  = GSTA_in_series[GSTA_in_mask_2023_2025 ]

ax1 = fig2.add_subplot(1,1,1)
ax1.set_ylim(-3, 3)
ax1.set_yticks([ -3, -2, -1, 0, 1, 2, 3])
ax1.axhline(0, color='BLACK', linewidth=1)
ax1.axhline(0.5, color='GRAY', linewidth=1.5)
ax1.axhline(-0.5, color='GRAY', linewidth=1.5)
ax1.plot(filtered_nino34_2023_2025.index, filtered_nino34_2023_2025.values,'ko-', color = 'blue', label = 'Nino3.4 anomaly')
ax1.set_ylabel('Nino3.4', c='blue')
ax1.set_xlabel('Time')
ax2 = ax1.twinx()
ax2.set_ylim(-2.5, 2.5)
ax2.set_yticks([-2.5, -1, 0, 1,1.5,2.5])
ax2.plot(filtered_GSTA_in_2023_2025.index, filtered_GSTA_in_2023_2025.values, 'rs-',color = 'r', label = 'Temperature anomaly')
ax2.set_ylabel('Temperature', c='r')

plt.suptitle('Monthly Nino3.4 and Global Mean Surface Temperature anomaly Time serise(2023/1-2024/3)')
plt.savefig('Monthly Nino3.4 and Global Mean Surface Temperature anomaly Time serise(20231-20243).jpg', dpi=300, bbox_inches='tight')
plt.show()


# plot the time series Nino3.4 and GSTA exclude Nino regions
fig3 = plt.figure(figsize=(12,10))

ax1 = fig3.add_subplot(3,1,1)
ax1.set_ylim(-3, 3)
ax1.set_yticks([ -3, -2, -1, 0, 1, 2, 3])
ax1.axhline(0, color='BLACK', linewidth=1)
ax1.axhline(0.5, color='GRAY', linewidth=1.5)
ax1.axhline(-0.5, color='GRAY', linewidth=1.5)
ax1.plot(filtered_nino34_1950_1975.index, filtered_nino34_1950_1975.values, color = 'blue', label = 'Nino3.4 anomaly')
ax1.set_ylabel('Nino3.4', c='blue')
ax4 = ax1.twinx()
ax4.set_ylim(-1, 1)
ax4.set_yticks([-1, -0.5, 0, 0.5, 1])
ax4.plot(filtered_GSTA_ex_1950_1975.index, filtered_GSTA_ex_1950_1975.values, color = 'r', label = 'GSTA (excluding Nino regions)')
ax4.set_ylabel('Temperature', c='r')

ax2 = fig3.add_subplot(3,1,2)
ax2.set_ylim(-3, 3)
ax2.set_yticks([ -3, -2, -1, 0, 1, 2, 3])
ax2.axhline(0, color='black', linewidth=1)
ax2.axhline(0.5, color='GRAY', linewidth=1.5)
ax2.axhline(-0.5, color='GRAY', linewidth=1.5)
ax2.plot(filtered_nino34_1975_2000.index, filtered_nino34_1975_2000.values, color = 'blue')
ax2.set_ylabel('Nino3.4', c='blue')
ax5 = ax2.twinx()
ax5.set_ylim(-1.5, 1.5)
ax5.set_yticks([-1.5,-0.75, 0, 0.75,1.5])
ax5.plot(filtered_GSTA_ex_1975_2000.index, filtered_GSTA_ex_1975_2000.values,  color = 'r')
ax5.set_ylabel('Temperature', c='r')

ax3 = fig3.add_subplot(3,1,3)
ax3.set_ylim(-3, 3)
ax3.set_yticks([ -3, -2, -1, 0, 1, 2, 3])
ax3.axhline(0, color='black', linewidth=1)
ax3.axhline(0.5, color='GRAY', linewidth=1.5)
ax3.axhline(-0.5, color='GRAY', linewidth=1.5)
ax3.plot(filtered_nino34_2000_2025.index, filtered_nino34_2000_2025.values, color = 'blue')
ax3.set_ylabel('Nino3.4', c='blue')
ax3.set_xlabel('Time(year)')
ax6 = ax3.twinx()
ax6.set_ylim(-2.5, 2.5)
ax6.set_yticks([-2.5,-1.5, -1, 0, 1.5,2.5])
ax6.plot(filtered_GSTA_ex_2000_2025.index, filtered_GSTA_ex_2000_2025.values, color = 'r')
ax6.set_ylabel('Temperature', c='r')

fig3.legend()
plt.suptitle('Monthly Nino3.4 and GSTA (excluding Nino regions) Time serise (1950-2024)')
plt.tight_layout() 
plt.savefig('Monthly Nino3.4 and GSTA Time serise excluding Nino regions.jpg', dpi=300, bbox_inches='tight')
plt.show()


# plot the time series include and exclude Nino regions
fig4 = plt.figure(figsize=(14,12))

ax1 = fig4.add_subplot(3,1,1)
ax1.set_ylim(-1, 1)
ax1.set_yticks([-1, -0.5,0,0.5, 1])
ax1.axhline(0, color='BLACK', linewidth=1)
ax1.plot(filtered_GSTA_in_1950_1975.index, filtered_GSTA_in_1950_1975.values, color = 'skyblue', label = 'Temperature anomaly')
ax1.set_ylabel('Temperature anomaly(°C)')
ax1.plot(filtered_GSTA_ex_1950_1975.index, filtered_GSTA_ex_1950_1975.values, color = 'r', label = 'Temperature anomaly exclude Nino regions')


ax2 = fig4.add_subplot(3,1,2)
ax2.set_ylim(-1, 1.5)
ax2.set_yticks([ -1, -0.5,0,0.5, 1, 1.5])
ax2.axhline(0, color='black', linewidth=1)
ax2.plot(filtered_GSTA_in_1975_2000.index, filtered_GSTA_in_1975_2000.values, color = 'skyblue')
ax2.set_ylabel('Temperature anomaly(°C)')
ax2.plot(filtered_GSTA_ex_1975_2000.index, filtered_GSTA_ex_1975_2000.values,  color = 'r')


ax3 = fig4.add_subplot(3,1,3)
ax3.set_ylim(-0.5, 2.1)
ax3.set_yticks([ -0.5,0,0.5, 1, 1.5, 2.1])
ax3.axhline(0, color='black', linewidth=1)
ax3.plot(filtered_GSTA_in_2000_2025.index, filtered_GSTA_in_2000_2025.values, color = 'skyblue')
ax3.set_ylabel('Temperature anomaly(°C)')
ax3.set_xlabel('Time(year)')
ax3.plot(filtered_GSTA_ex_2000_2025.index, filtered_GSTA_ex_2000_2025.values, color = 'r')

fig4.legend()
plt.suptitle('Monthly Global Mean Surface Average Temperature Anomaly Time serise')
plt.tight_layout() 
plt.savefig('Monthly Global Mean Surface Average Temperature Anomaly Time serise.jpg', dpi=300, bbox_inches='tight')
plt.show()

#######END######


