import requests
import json
from time import sleep
import os

endpoint = "https://opendata-download-lightning.smhi.se/api.json"
version = "latest"

year_min = 2012
year_max = 2021
wait_time_s = 10

curr_dir = os.curdir
data_dir = curr_dir + '/../data/'
print(data_dir)

r = requests.get(endpoint)

#print(r.text)

data = json.loads(r.text)

print(type(data))

#Get link for latest json version
for i in range(len(data['version'])):
    ver = data['version'][i]
    if ver['key'] == version:
        print(ver['key'])
        for link in ver['link']:
            if link['rel'] == 'version' and link['type'] == 'application/json':
                print(link['type'])
                href = link['href']

years = []
months = []
days = []
file_links = []

#Get link for years
r_year = requests.get(href)
data_year = json.loads(r_year.text)

for year in data_year['resource']:
    if year_min <= int(year['title']) <= year_max:
        year_dict = {'key': year['key'], 'title': year['title']}
        print(year_dict['title'])
        for link in year['link']:
            if link['rel'] == 'year' and link['type'] == 'application/json':
                year_dict['href'] = link['href']
                years.append(year_dict)

        #Get link for months
        r_month = requests.get(year_dict['href'])
        data_month = json.loads(r_month.text)


        for month in data_month['month']:
            month_dict = {'key': month['key'], 'title': month['title']}

            for link in month['link']:
                if link['rel'] == 'month' and link['type'] == 'application/json':
                    month_dict['href'] = link['href']
                    months.append(month_dict)

            #Get link for days
            r_day = requests.get(month_dict['href'])
            data_day = json.loads(r_day.text)

            for day in data_day['day']:
                day_dict = {'key': day['key'], 'title': day['title']}

                for link in day['link']:
                    if link['rel'] == 'day' and link['type'] == 'application/json':
                        day_dict['href'] = link['href']
                        days.append(day_dict)
                        print(day_dict)
                #Get data
                r_file_info  = requests.get(day_dict['href'])
                data_file_info = json.loads(r_file_info.text)

                file_info = data_file_info['data'][0]
                file_info_dict = {'key': file_info['key'], 'title': file_info['title']}

                for link in file_info['link']:
                    if link['rel'] == 'data' and link['type'] == 'application/json':
                        file_info_dict['href'] = link['href']
                        file_links.append(file_info_dict)

                        sleep(wait_time_s)
                        r_file = requests.get(file_info_dict['href'])
                        print(r_file)
                        data_file = r_file.json()
                        values = data_file['values']
                        with open('test.txt','w') as test:
                            test.write(str(data_file))

                        #Check if directory exists, and create if not
                        directory = data_dir  + year_dict['title'] + '/'   
                        if not(os.path.isdir(directory)): 
                            os.makedirs(directory)
                        directory = directory + month_dict['title'] + '/'
                        if not(os.path.isdir(directory)): 
                            os.makedirs(directory)

                        #Save data to file
                        with open(directory + file_info_dict['title'] + '.json','w') as json_file:
                            json.dump(data_file,json_file)
