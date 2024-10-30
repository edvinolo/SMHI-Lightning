import matplotlib.pyplot as plt
import numpy as np
import shapefile as shp
import pandas as pd
import time
from sklearn.metrics import pairwise_distances_chunked
from scipy.linalg import norm

def read_shapefile(sf):
    #Converting shapefile data to pandas datafram
    #from: https://towardsdatascience.com/mapping-with-matplotlib-pandas-geopandas-and-basemap-in-python-d11b57ab5dac

    #fetching the headings from the shape file
    fields = [x[0] for x in sf.fields][1:]
    #fetching the records from the shape file
    records = [list(i) for i in sf.records()]
    shps = [s.points for s in sf.shapes()]
    #converting shapefile data into pandas dataframe
    df = pd.DataFrame(columns=fields, data=records)
    #assigning the coordinates
    df = df.assign(coords=shps)
    return df
 
def dist_func(x,y):
    return norm(x-y)


def reduce_dist(d_chunk,start):
    nbh_th = 1.0
    #neighbors = np.count_nonzero(np.less(d_chunk,nbh_th),axis=1)-1
    neighbors = np.sum(np.less(d_chunk,nbh_th),axis=1)
    print(d_chunk.shape)
    print(neighbors.size)
    print(d_chunk.dtype)
    return neighbors

def array_trickery(lon,lat,both,nbh_th = 1.0):
    print('Doing the array trickery stuff...')
    t_1 = time.perf_counter()
    
    neighbors_2d = np.zeros(both.size)
    slice_index = 0

    combined = np.column_stack((lon,lat))


    chunks = pairwise_distances_chunked(combined, reduce_func = reduce_dist,n_jobs = 1)
    for chunk in chunks:
        temp_size = chunk.size
        slice_index += temp_size
        print(slice_index)
        neighbors_2d[slice_index-temp_size:slice_index] = chunk
    neighbors_2d -= 1
    t_2 = time.perf_counter()
    time_2d = t_2-t_1

    print('Doing the loop stuff...')
    neighbors = []
    t_1 = time.perf_counter()
    for event in both:
        dist = np.abs(both-event)
        n = np.count_nonzero(np.less(dist,nbh_th))-1
        neighbors.append(n)
    t_2 = time.perf_counter()
    time_loop = t_2-t_1
    max_error = np.max(neighbors_2d-np.array(neighbors))

    print(f'Loop time: {time_loop} s')
    print(f'Biggest difference between loop and chunks: {max_error}')
    print(combined.dtype)
    print(dist.dtype)

    fig_bug, ax_bug = plt.subplots()
    ax_bug.plot(neighbors_2d-np.array(neighbors))
    
    return [neighbors_2d,time_2d]

def neighbor_map(both,data_dir):
    min_lon_map = -15
    max_lon_map = 40
    nbh_th = 1.0

    border_dir = f'{data_dir}map_data/ne_boundary_lines/'
    country_dir = f'{data_dir}map_data/ne_countries/'
    coast_dir = f'{data_dir}map_data/ne_coastline/'


    lon  = np.real(both)
    lat = np.imag(both)
    neighbors, time = array_trickery(lon,lat,both,nbh_th)

    print(f'Time to find neighbors: {time} s')

    
    fig_loc,ax_loc = plt.subplots()
    sc_loc = ax_loc.scatter(lon,lat,c = neighbors,cmap='inferno')
    ax_loc.set_xlim([min_lon_map,max_lon_map])
    cb_loc = fig_loc.colorbar(sc_loc)

    #load and plot border data
    border_path = f'{border_dir}ne_10m_admin_0_boundary_lines_land.shp'
    sf = shp.Reader(border_path)

    country_path = f'{country_dir}ne_10m_admin_0_countries.shx'
    sf_country = shp.Reader(country_path)

    df = read_shapefile(sf)
    df_country = read_shapefile(sf_country)

    temp = np.array(df_country[df_country['SOVEREIGNT'] == 'Sweden']['coords'])
    coords = temp[0]
    lon_data = [x[0] for x in coords]
    lat_data = [x[1] for x in coords]
    ax_loc.plot(lon_data,lat_data,'yx')

    return 0
