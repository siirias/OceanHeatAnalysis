import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
import requests



# 'world_sst'
# 'n_atlantic_sst'
datasets = ['world_sst', 'n_atlantic_sst']
y_limits_anomaly = [-1.0, 1.5]
y_limits = [18.0, 25.0]
figure_size = tuple(np.array((10,4))*1.0)
out_directory = "c:/Data/global/"
fig_dpi = 150
for dataset in datasets:
    # # Load data from the online source

    if( dataset == 'world_sst'):
        variable_text = 'SST'
        area_text = 'World'
        unit_text = '°C'
        url = 'https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_world2_sst_day.json'
        response = requests.get(url)
        data = response.json()
        
    if( dataset == 'n_atlantic_sst'):
        variable_text = 'SST'
        area_text = 'North Atlantic'
        unit_text = '°C'
        url = 'https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_natlan1_sst_day.json'
        response = requests.get(url)
        data = response.json()

    
    mean_temperatures = np.zeros(365)
    
    # Count the number of years in the range 1986 to 2016
    count = 0
    
    
    # Start and end years for color mapping
    start_year = None
    end_year = None
    parsed_data = {}
    
    # Parse each year to find start and end year
    for year_data in data:
        year_name = year_data['name']
        try:
            year = int(year_name)
            if start_year is None or year < start_year:
                start_year = year
            if end_year is None or year > end_year:
                end_year = year
        except ValueError:
            if "mean" in year_name:
                mean_temperatures = np.array(year_data['data'][:365])
                mean_label = year_name
            if "minus" in year_name:
                sigma2_lower_limit = np.array(year_data['data'][:365])
                sigma_lower_label = year_name
            if "plus" in year_name:
                sigma2_upper_limit = np.array(year_data['data'][:365])
                sigma_upper_label = year_name
    
    # Re-parse each year for plotting
    for year_data in data:
        year_name = year_data['name']
        # Check if the year_name can be converted to an integer
        try:
            year = int(year_name)
        except ValueError:
            continue
    
        # Consider only first 365 days
        temperatures = year_data['data'][:365] 
        # Convert temperatures to a masked array where None values are masked
        the_mask = [not isinstance(i,float) for i in temperatures]
        temperatures = np.ma.masked_where(the_mask, temperatures)
        temperatures[temperatures.mask] = np.nan
        parsed_data[year] = temperatures
    #     if 1986 <= year <= 2016:
    #         count += 1
    #         mean_temperatures = mean_temperatures + temperatures.filled(0)
    
    
    # # Calculate mean temperatures
    # mean_temperatures = mean_temperatures / count if count > 0 else mean_temperatures
    
    for plot_anomaly in [True, False]:
        # Setup colormap
        plt.figure(figsize = figure_size)
        cmap = plt.get_cmap('coolwarm')
        norm = plt.Normalize(start_year, end_year)
        
        for year in parsed_data.keys():
            temperatures = parsed_data[year]
            # Make all years except the last two transparent
            alpha = 1 if year >= end_year - 1 else 0.5
            if not plot_anomaly:
                plt.plot(range(1, len(temperatures) + 1), temperatures, 
                         color=cmap(norm(year)), alpha = alpha)
            else:
                plt.plot(range(1, len(temperatures) + 1), temperatures- mean_temperatures, 
                         color=cmap(norm(year)), alpha = alpha)
        
        
        
        # Add mean temperatures to plot
        if plot_anomaly:
            to_minus = mean_temperatures
        else:
            to_minus = mean_temperatures*0.0
        if(not plot_anomaly):
            plt.plot(range(1, 366), mean_temperatures - to_minus, 
                     label=mean_label, linewidth=1.5, 
                     linestyle='--', color='black', alpha = 0.6)
        plt.plot(range(1, 366), sigma2_upper_limit - to_minus, 
                 label=sigma_upper_label, 
                 linewidth=1, linestyle='--', color='black', alpha = 0.6)
        plt.plot(range(1, 366), sigma2_lower_limit - to_minus, 
                 label=sigma_lower_label, 
                 linewidth=1, linestyle='--', color='black', alpha = 0.6)
    
        if plot_anomaly:
            plt.title(f'{variable_text} Anomaly of {area_text} Over Time\n')
        else:
            plt.title(f'{variable_text} of {area_text} Over Time\n')
        plt.xlabel('Day of the Year')
        plt.ylabel(f'{variable_text} Anomaly ({unit_text})' if plot_anomaly else f'{variable_text} ({unit_text})')
        
        # add the refernce to data source
        plt.annotate( f"data from: {url}", (0.5,1.05), 
                     xycoords = 'axes fraction', 
                     horizontalalignment = 'center', 
                     verticalalignment = 'top', fontsize = 8)
        if plot_anomaly:
            plt.ylim(y_limits_anomaly)
        else:
            plt.ylim(y_limits)
        plt.xlim((0,365))
        # Create colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Year')
        
        plt.legend()
        
        plt.grid(True)
        plt.show()
        if(plot_anomaly):
            extra = 'anomaly'
        else:
            extra = ''
        filename = f"{variable_text}_{area_text}_{extra}.png"
        plt.savefig(out_directory + filename+'.png' ,\
                    facecolor='w',dpi=fig_dpi,bbox_inches='tight')
