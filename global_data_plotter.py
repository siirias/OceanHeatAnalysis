import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cm import ScalarMappable
import requests
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.colors as mcolors

# 'world_sst'
# 'n_atlantic_sst'
datasets = ['world_sst', 'n_atlantic_sst']
y_limits_anomaly = [-1.0, 1.5]
y_limits = [18.0, 25.0]
figure_size = tuple(np.array((10,4))*1.0)
out_directory = "c:/Data/global/"
fig_dpi = 600
plot_limits = False
plot_average = True
cloud_alpha = 0.6
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

    # Gather the data as one timeline for other plots
    # Create an empty DataFrame
    df = pd.DataFrame()
    # Loop over the years in the dictionary
    for year, values in parsed_data.items():
#        if type(values.mask) == np.bool_ or np.sum(list(map(np.isnan,values))) < 7: #don't use years with lots of data missing
            # Create a date range for the year
            dates = pd.date_range(start=f'{year}-01-01', periods=len(values), freq='D')
        
            # Create a DataFrame for the year
            year_df = pd.DataFrame({
                'date': dates,
                'value': values
            })
            # Append the year DataFrame to the main DataFrame
            df = pd.concat((df,year_df))
    df.set_index('date', inplace=True)
    
    # set the anomaly in the dataframe too:
    # Create a new column 'day_of_year' in the dataframe
    df['day_of_year'] = df.index.dayofyear
    
    # Subtract the corresponding mean temperature from the original value
    df['anomaly'] = df['value'] - df['day_of_year'].apply(lambda x: mean_temperatures[x - 1])
    
    for plot_type in ['sst_anomaly', 'sst', 'trend', 'histogram']: 
        # Setup colormap
        anomaly_plot = False        

        plt.figure(figsize = figure_size)
        cmap = plt.get_cmap('coolwarm') #coolwarm
        norm = plt.Normalize(start_year, end_year)
        #plot type specific parts here
        if plot_type in ['sst_anomaly','sst']:
            if plot_type in ['sst_anomaly']:
                anomaly_plot = True
            for year in parsed_data.keys():
                temperatures = parsed_data[year]
                # Make all years except the last two transparent
                alpha = 1 if year >= end_year - 1 else cloud_alpha
                the_color = cmap(norm(year))
                the_label = None
                the_width = 0.7
                if(year == end_year):
                    the_color = cmap(norm(year))
                    the_label = year
                    the_width = 1.3
                if(year == end_year - 1):
                    the_color = (0.5,0.25,0.25)
                    the_label = year
                    the_width = 1.3
                if plot_type == 'sst':
                    main_data = temperatures
                elif plot_type == 'sst_anomaly':
                    main_data = temperatures - mean_temperatures
                else:
                    main_data = temperatures
                plt.plot(range(1, len(temperatures) + 1), main_data, 
                         label = the_label, 
                         color = the_color, alpha = alpha, 
                         linewidth = the_width)
            
            
            # Add mean temperatures to plot
            if anomaly_plot:
                to_minus = mean_temperatures
            else:
                to_minus = mean_temperatures*0.0
            if(not anomaly_plot):
                if(plot_average):
                    plt.plot(range(1, 366), mean_temperatures - to_minus, 
                             label=mean_label, linewidth=1.5, 
                             linestyle='--', color='black', alpha = 0.6)
            if(plot_limits):
                plt.plot(range(1, 366), sigma2_upper_limit - to_minus, 
                         label=sigma_upper_label, 
                         linewidth=1, linestyle='--', color='black', alpha = 0.6)
                plt.plot(range(1, 366), sigma2_lower_limit - to_minus, 
                         label=sigma_lower_label, 
                         linewidth=1, linestyle='--', color='black', alpha = 0.6)
        
            if plot_type == 'sst_anomaly':
                plt.title(f'{variable_text} Anomaly of {area_text} Over Time\n')
                plt.ylabel(f'{variable_text} Anomaly ({unit_text})') 
            else:
                plt.title(f'{variable_text} of {area_text} Over Time\n')
                plt.ylabel(f'{variable_text} ({unit_text})')
                
            plt.xlabel('Day of the Year')

            # Create colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Year')
            cbar.set_alpha(cloud_alpha)
            cbar.draw_all()  # Redraw the colorbar with the modified alpha value        

            if anomaly_plot:
                plt.ylim(y_limits_anomaly)
            else:
                plt.ylim(y_limits)
            plt.xlim((0,365))
            plt.gca().xaxis.set_major_locator(MonthLocator())
            plt.gca().xaxis.set_major_formatter(DateFormatter("%b"))
            plt.legend()
            
            plt.grid(True)

        elif plot_type == 'trend':

            yearly_mean = df.value.resample('Y').mean()[1:-1]
            yearly_std = df.value.resample('Y').std()[1:-1]
            # After calculating yearly_mean and yearly_std
            # Reset the index to convert the dates from the index to a column
            yearly_mean = yearly_mean.reset_index()
            yearly_std = yearly_std.reset_index()
            
            # Convert the dates to year only for plotting
            yearly_mean['date'] = yearly_mean['date'].dt.year
            yearly_std['date'] = yearly_std['date'].dt.year
            
            #fit polynomial
            n = 1
            fit_time = yearly_mean['date'] - yearly_mean['date'][0]
            coeffs = np.polyfit(fit_time, yearly_mean['value'], n)
            poly = np.poly1d(coeffs)
            y_fit = poly(fit_time)
            the_label = "Fit: °C/decade: " + ", ".join(['{:0.4f}'.format(i*10.0) for i in poly.coefficients[:-1]][::-1])
            # Plot the means with error bars for the standard deviation
            plt.plot(yearly_mean['date'], yearly_mean['value'], 'o')
            plt.plot(yearly_mean['date'], y_fit, '-', label=the_label)  # fitted polynomial
            
            # Set the labels for the x and y axes
            plt.xlabel('Year')
            plt.ylabel('Mean SST per year')
            plt.title(f'Trend of {variable_text} of {area_text} Over Time\n')
            plt.legend()
            plt.grid(True)
        elif plot_type == 'histogram':
            n = 10
            bins = 40 #50
            trigger_dT = 0.35
            df_diff = df.diff(periods = n)
            flat_data = df_diff.anomaly.values.flatten()
            flat_data = np.array([x for x in flat_data if not np.isnan(x)])
            # Create weights for each data point to convert counts into percentages
            weights = 100.0 * np.ones_like(flat_data) / len(flat_data)  # 100* to get percentages
            counts, edges = np.histogram(flat_data, bins=bins) # To get max height.
            
            
            # Plot the histogram
            plt.hist(flat_data, bins=bins, weights = weights, color = (0.6, 0.6, 0.8), edgecolor='black', zorder = 2)
            plt.xlabel('Difference')
            plt.ylabel('Percentage')
            plt.title(f'Histogram of {variable_text} Anomaly differences over {n} days in {area_text}\n')
            plt.gca().yaxis.set_major_locator(
                MultipleLocator(100.0*np.round((max(counts) / len(flat_data))/10,3)))
            plt.gca().xaxis.set_major_locator(
                MultipleLocator(np.round((flat_data.max() - flat_data.min())/15.0,2)))
            plt.grid(True, alpha = 0.3)
            
            # Compute the percentage of cases where absolute difference is over x
            over_dT = np.sum(np.abs(flat_data) > trigger_dT)
            percentage_over_dT = over_dT / len(flat_data)
            
            # Create a text box with the statistics
            x = df_diff.anomaly
            stats_text = f'Datapoints: {float(x.count()):.0f}\n'+ \
                        f'{bins} bins\n'+ \
                        f'Mean: {float(x.mean()):.2f}\n'+ \
                        f'Std: {float(x.std()):.2f}\n'+ \
                        f'Min: {float(x.min()):.2f}\n'+ \
                        f'Max: {float(x.max()):.2f}\n'+ \
                        f'dT over {trigger_dT:.2f}: {percentage_over_dT:0.1%}'
            plt.text(0.02, 0.96, stats_text, 
                     transform=plt.gca().transAxes, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            # Some visual eye candy:
            ax = plt.gca()
            plt.xlim((flat_data.min(), flat_data.max()))
            # Create a colormap.
            cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
            
            # Create a norm from the bounds of your data.
            range_lim = np.max([np.abs(flat_data.min()), np.abs(flat_data.max())])
            norm = mcolors.Normalize(vmin=-range_lim, vmax=range_lim)
            
            # Create a 2D array with the x and y extents of your data.
            img = np.array([[norm(v) for v in np.linspace(flat_data.min(), flat_data.max(), 100)]])
            img = np.vstack((img, img))
            
            # Display the colormap image with correct extents, but make it fit the Y scale.
            plt.imshow(img, cmap=cmap, aspect='auto', extent=[-range_lim, 
                        range_lim, ax.get_ylim()[0], ax.get_ylim()[1]], 
                       alpha = 0.4, zorder=1)


            
            plt.show()            
            
        # add the refernce to data source
        plt.annotate( f"data from: {url}", (0.5,1.0), 
                     xycoords = 'axes fraction', 
                     horizontalalignment = 'center', 
                     verticalalignment = 'bottom', fontsize = 9)
        plt.show()
        if plot_type == 'sst_anomaly':
            extra = '_anomaly'
        elif plot_type == 'trend':
            extra = '_trend'
        elif plot_type == 'sst':
            extra = ''
        elif plot_type == 'histogram':
            extra = 'histogram'
        filename = f"{variable_text}_{area_text}{extra}"
        plt.savefig(out_directory + filename+'.png' ,\
                    facecolor='w',dpi=fig_dpi,bbox_inches='tight')
