"""
Script to download netcdf from TRMM website to s3 and optionally retrieve from s3.
"""
import requests
from bs4 import BeautifulSoup
import os
from os.path import expanduser
import time
import logging
import boto3
import json
import multiprocessing
from multiprocessing import Pool
import itertools
import sys
import argparse


class TRMM_data(object):
    """
        Class to access data from TRMM website and upload to S3.
        : param month_year(tuple containing year and month combination.)
    """
    def __init__(self, month_year):
        self.month_year = month_year
        self.creds_data = {}
        self.auth_data = ('pshivraj@uw.edu', 'pshivraj@uw.edu')
        self.client = None
        self.output_folder = 'EPO/'

    def load_creds(self):
        """
            Utility function to read s3 credential file for
            data upload to s3 bucket.
        """
        home = expanduser("~")
        with open(os.path.join(home, 'creds.json')) as creds_file:
            self.creds_data = json.load(creds_file)

    def data_fetch_netcdf(self):
        """
            Utility function to upload netcdf file from
            TRMM website to s3 partitioned by year_month.
        """
        self.client = boto3.client('s3', aws_access_key_id=self.creds_data['key_id'],
                    aws_secret_access_key=self.creds_data['key_access'])
        year = self.month_year[0]
        month = self.month_year[1]
        # change output folder to desired location from TRMM website
        # folder structure to partitioned the data year_month
        output_temp = self.output_folder + year + '_' + month
        url_data = "http://trmm.atmos.washington.edu/{}interp_data/{}/{}".format(self.output_folder, year, month)
        print(url_data)
        start_time_year_month = time.time()
        r = requests.get(url_data, auth=self.auth_data)
        # check if url exists then extract netcdf links to download and upload to s3.
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, features='lxml')
            for link in soup.findAll('a'):
                link_url = link.get('href')
                write_path = os.path.join(output_temp, link_url)
                if link_url.endswith('.nc4'):
                    file_url = url_data + '/' + link_url
                    r = requests.get(file_url, auth=self.auth_data, stream=True)
                    if r.status_code == 200:
                        self.client.put_object(Body=r.content, Bucket='himatdata', Key='Trmm/' + write_path)
            logging.info("Done with Year Month: %s", month_year)
            print("--- %s seconds ---" % (time.time() - start_time_year_month))

        else:
            print('No data/authentication for'.format(month_year))

    def data_pull_s3(self):
        """
            Utility function to get data from s3 partitioned by year_month.
        """
        year = self.month_year[0]
        month = self.month_year[1]
        self.s3 = boto3.resource('s3',aws_access_key_id=self.creds_data['key_id'],
                                aws_secret_access_key=self.creds_data['key_access'])
        bucket = self.s3.Bucket('himatdata')
        home = os.getcwd()
        file_path = os.path.join(*[home, 'Trmm/', self.output_folder, year + '_' + month])
        print(file_path)
        if not os.path.exists(file_path):
           os.makedirs(file_path)
        for obj in bucket.objects.filter(Delimiter='', Prefix='Trmm/{}{}_{}'.format(self.output_folder, year, month)):
            if obj.key.endswith('.nc4'):
                bucket.download_file(obj.key,os.path.join(os.path.join(home, obj.key)))
        logging.info("Done with Year Month: %s", month_year)


def _multiprocess_handler(month_year):
     """
        Utility function pass to multiprocessing pool
     """
     trmm_data = TRMM_data(month_year)
     trmm_data.load_creds()
     if args.type == 's3_push':
         trmm_data.data_fetch_netcdf()
     elif args.type == 's3_pull':
         trmm_data.data_pull_s3()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to push or retrieve data from s3')
    parser.add_argument('-t', '--type', choices=['s3_pull', 's3_push'], default="s3_pull", help='type of operation on data')
    args = parser.parse_args()
    logging.basicConfig(filename='trmm_{}.log'.format(args.type), level=logging.INFO)
    logging.info("Executing %s", args.type)
    # define month and year to get data for
    months = [str(i).zfill(2) for i in range(1, 13)]
    years = [str(i).zfill(4) for i in range(1998, 2014)]
    # generate combination of year and month.
    month_year = list(itertools.product(years, months))
    start_time = time.time()
    # multiprocess the file upload for various year and month combinations.
    process = 15
    process = multiprocessing.cpu_count() if process > multiprocessing.cpu_count() else process
    pool = Pool(process)
    pool.map(_multiprocess_handler, month_year, chunksize=len(month_year)//process)
    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))
