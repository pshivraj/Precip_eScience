import requests
from bs4 import BeautifulSoup
import os

def data_fetch_netcdf(user_login, url, output_folder):
    r = requests.get(url_date, auth = auth_data)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text)
        url = []
        for link in soup.findAll('a'):
            link_url = link.get('href')
            current_dir = os.getcwd()
            write_path = os.path.join(os.path.sep, current_dir, output_folder, link_url)
            if link_url.endswith('.nc4'):
                file_url = url_date + '/' + link_url
                print(file_url)
                r = requests.get(file_url, auth = auth_data,stream = True)
                print(r.status_code)
                if r.status_code == 200:
                    with open(write_path, 'wb') as f:
                        f.write(r.content)
    else:
        print('No data for'.format(url))
            
if __name__=='__main__':
    start_time = time.time()
    auth_data = ('username','username')
    # define month and year to get data for
    months = [str(i).zfill(2) for i in range(1, 13)]
    years = [str(i).zfill(4) for i in range(1998, 2018)]

    output_folder = 'raw_files/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for year in years:
        for month in months:
            url_date = "http://trmm.atmos.washington.edu/EPO/interp_data/{}/{}".format(year,month)
            output_temp = output_folder + year + '_' + month
            if not os.path.exists(output_temp):
                os.makedirs(output_temp)
            data_fetch_netcdf(auth_data, url_date,output_temp)
    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))