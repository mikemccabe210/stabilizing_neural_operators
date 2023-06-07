import cdsapi
import numpy as np
import os

base_path = '/project/projectdirs/dasrepo/ERA5/wind_levels/6hr/msp_mslp' 
if not os.path.isdir(base_path):
    os.makedirs(base_path)
years = np.arange(1979, 2021)  
 
c = cdsapi.Client()

 
for year in years:
    
    year_str = str(year) 
    file_str = base_path + '/sp_mslp_' + year_str  + '.nc'
    print(year_str)
    print(file_str)
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'mean_sea_level_pressure', 'surface_pressure',
            ],          
            'year': year_str,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00','18:00',
            ],          
        },
        file_str)

