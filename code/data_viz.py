import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import shapefile as shp
import pandas as pd
import time
from sympy.ntheory import factorint
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neighbor_map import neighbor_map
from cloud_classifier import cloud_classifier


def main():
    year = 2021
    month_start = 1
    month_stop = 12
    min_lon_map = -15
    max_lon_map = 40
    nbh_th = 2
    #month = 7
    #day = 24

    curr_dir = os.curdir
    data_dir = curr_dir + '/../data/'
    print(data_dir)
    year_dir = f'{data_dir}{year}/'
    border_dir = f'{data_dir}map_data/ne_boundary_lines/'
    country_dir = f'{data_dir}map_data/ne_countries/'
    coast_dir = f'{data_dir}map_data/ne_coastline/'

    strikes = []
    days = []
    lat = []
    lon = []
    both = []
    peak_current = []
    cloud_indicator = []
    rise_time = []
    peak_to_zero = []
    max_rise_rate = []
    multiplicity = []
    neighbors = []
    day = 1
    for month in range(month_start,month_stop+1):
        if month < 10:
            month = f'0{month}'
        else:
            month = str(month)
        month_dir = f'{year_dir}{year}-{month}/'

        for file in sorted(os.listdir(month_dir)):
            print('-------------------------------')
            print(file)
            file = f'{month_dir}/{file}'
            with open(file,'r') as data_file:
                data = json.load(data_file)
                values = data['values']
                no_of_srikes = len(values)
                print(no_of_srikes)
                print('-------------------------------')
                strikes.append(no_of_srikes)
                days.append(day)
                day +=1
                if no_of_srikes > 0:
                    for strike in values:
                        la = float(strike['lat'])
                        lo =float(strike['lon'])
                        peak = float(strike['peakCurrent'])
                        cloud = int(strike['cloudIndicator'])

                        lat.append(la)
                        lon.append(lo)
                        both.append(lo + 1j*la)
                        peak_current.append(peak)
                        cloud_indicator.append(cloud)
                        rise_time.append(float(strike['riseTime']))
                        peak_to_zero.append(float(strike['peakToZeroTime']))
                        max_rise_rate.append(float(strike['maxRateOfRise']))
                        multiplicity.append(int(strike['multiplicity']))

    both = np.array(both)
    print(f'The number of events: {both.size}')


    if sys.argv[1] == '-nbh_map':
        neighbor_map(both,data_dir)
    elif sys.argv[1] == '-cloud':
        #Trying out logistic regression to predict cloud-to-cloud or cloud-to-ground strike
        X_data = np.column_stack((peak_current,rise_time,peak_to_zero,max_rise_rate,multiplicity))
        Y_data = np.array(cloud_indicator)

        cloud_classifier(X_data,Y_data)


    # #Trying out logistic regression to predict cloud-to-cloud or cloud-to-ground strike
    # X_data = np.column_stack((peak_current,rise_time,peak_to_zero,max_rise_rate,multiplicity))
    # Y_data = np.array(cloud_indicator)

    # #Split into training and testing data
    # X_train, X_test, Y_train, Y_test = train_test_split(X_data,Y_data,test_size = 0.25, random_state = 0)

    # #Normalize data (this step seem quite strange)
    # ss_train = StandardScaler()
    # X_train = ss_train.fit_transform(X_train)

    # ss_test = StandardScaler()
    # X_test = ss_test.fit_transform(X_test)

    # #Train classifier
    # logistic_classifier = LogisticRegression(random_state = 0)
    # logistic_classifier.fit(X_train,Y_train)

    # #Test classifier
    # predictions = logistic_classifier.predict(X_test)
    # cm = confusion_matrix(Y_test, predictions)
    # TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
    # print('True Positive(TP)  = ', TP)
    # print('False Positive(FP) = ', FP)
    # print('True Negative(TN)  = ', TN)
    # print('False Negative(FN) = ', FN)
    # accuracy =  (TP + TN) / (TP + FP + TN + FN)
    # print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))
    # score = logistic_classifier.score(X_test,Y_test)
    # print(f'Accuracy score: {score}')


    fig_sum,ax_sum = plt.subplots()
    ax_sum.plot(days,strikes)

    # fig_loc,ax_loc = plt.subplots()
    # sc_loc = ax_loc.scatter(lon,lat,c = neighbors,cmap='inferno')
    # ax_loc.set_xlim([min_lon_map,max_lon_map])
    # cb_loc = fig_loc.colorbar(sc_loc)

    # #load and plot border data
    # border_path = f'{border_dir}ne_10m_admin_0_boundary_lines_land.shp'
    # sf = shp.Reader(border_path)

    # country_path = f'{country_dir}ne_10m_admin_0_countries.shx'
    # sf_country = shp.Reader(country_path)

    # df = read_shapefile(sf)
    # df_country = read_shapefile(sf_country)

    # temp = np.array(df_country[df_country['SOVEREIGNT'] == 'Sweden']['coords'])
    # coords = temp[0]
    # lon_data = [x[0] for x in coords]
    # lat_data = [x[1] for x in coords]
    # ax_loc.plot(lon_data,lat_data,'yx')

    # fig_time, ax_time = plt.subplots()
    # ax_time.plot(size_list,time_list)
    # ax_time.axhline(time_loop)

    # fig_peak, ax_peak = plt.subplots()
    # ax_peak.plot(cloud_indicator,peak_current,'o')

    # fig_test, ax_test = plt.subplots()
    # sc_test = ax_test.scatter(rise_time,peak_to_zero,c = cloud_indicator,cmap = 'inferno')
    # cb_test = fig_test.colorbar(sc_test)

    plt.show()



if __name__ == '__main__':
    main()