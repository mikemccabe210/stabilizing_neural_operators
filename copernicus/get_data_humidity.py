import cdsapi
import numpy as np
import os

usr = 'p' # p, s 
base_path = '/project/projectdirs/dasrepo/ERA5/humidity/6hr/' + usr
if not os.path.isdir(base_path):
    os.makedirs(base_path)

year_dict = {'p': np.arange(1979,2000), 's' : np.arange(2000, 2021)}

years = year_dict[usr]  

c = cdsapi.Client()

    
for year in years:
    
    year_str = str(year) 
    pressure_str = '500_850' 
    file_str = base_path + '/rel_humidity_'+ pressure_str + '_'  + year_str  + '.nc'
    print(year_str)
    print(file_str)
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'pressure_level': ['500', '850',],
            'variable': [
                'relative_humidity',
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
