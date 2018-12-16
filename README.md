# Precipitation incubator at eScience winter 2018-2019

* [TRMM data](http:trmm.atmos.washington.edu) and all three researchers have been granted ftp server access.

## How to scan these data

* [TRMM](https://en.wikipedia.org/wiki/Tropical_Rainfall_Measuring_Mission) launched on 27 Nov 1997
  * TRMM is in LEO with 35 degrees inclination, 92.5 minutes / orbit. Example data below is from orbit 6294; Jan 1 1999
* 20 or so regions... some overlap; 17 years x 12 months per year x 30 days per month x 130 files per day x 70kb per file ~ 1.1TB
* **interp_data** folder only
  * XYZ is a region code, e.g. EPO/ = East Pacific Ocean (so 20 or so of these)
  * Let yyyy be the year and mm be the month running from 1998 01 to 2014 08 so '2003/' and '04/' for example
  * The URL directory format is then 'http://trmm.atmos.washington.edu/' + TLA + 'interp_data/' + yyyy + mm
* Files cover short time intervals
  * ```TPR7_uw1_06294.19990101.002101_EPO.nc4``` is our example filename; notice orbit, date, time and region


### Why copy the data? What is S3?


S3 is AWS object storage. It does not behave like a UNIX file system in that one can't do random access into the bytes of a file. No matter; treating the files as hermetic objects will work and S3 access is reasonably fast; and we can work with it pretty much as though it is a file system using Python, specifically pandas and/or xarray. We do want to make sure the S3 bucket and the Jupyter Hub are in the same region. 

I keep asking Shiv if there is some way of treating the UW Atmos TRMM ftp server like a virtual file system and he keeps patiently explaining NO so that's why I am suggesting working from a cloud copy. We can share the results with the *atmos* folks in case they are interested in our approach to data access. 

Shiv has the copy process running with only one login per directory; so it seems to be reasonably efficient and quick "per region-month". He can comment on wall clock time. If we take that to be 3 minutes then we have a total copy time of 3/60 * 20 * 17 * 12 = 220 hours or ten days. That's rather a lot of time but at least there is no cost associated to uploading data to AWS in this way.

@Shiv this would be a good time to confirm / fix my numbers.

#### code example by Shiv

Note that by default username and password are identical per atmos policy.

```
import requests
auth_data= ('username','password')
url = "http://trmm.atmos.washington.edu/EPO/interp_data/1999/01/TPR7_uw1_06294.19990101.002101_EPO.nc4"
r = requests.get(url, auth = auth_data,stream = True)
filename = 'test.nc4'
if r.status_code == 200:
    with open(filename, 'wb') as f:
        f.write(r.content)
```
