# Precipitation incubator at eScience winter 2018-2019

* [TRMM data](http:trmm.atmos.washington.edu) and all three researchers have been granted ftp server access.

## How to scan these data

* 20 or so regions?
* 17 years, 12 months per year, 30 days per month, 130 files per day, 70kb per file
* Total volume estimate: 1.1TB
* **interp_data** folder only
  * Let TLA be the region code, e.g. EPO/ = East Pacific Ocean (so 20 or so of these)
  * Let yyyy be the year and mm be the month running from 1998 10 to 2014 08 so 2003/ and 04/ for example
  * The URL directory format is then 'http://trmm.atmos.washington.edu/' + TLA + 'interp_data/' + yyyy + mm
* Each file covers a fairly short time range; need to decompose a data file or two (small / large) in a Notebook 


### Why copy the data? What is S3?

S3 is AWS object storage. It does not behave like a UNIX file system in that one can't do random access into the bytes of a file. No matter; treating the files as hermetic objects will work and S3 access is reasonably fast; and we can work with it pretty much as though it is a file system using Python, specifically pandas and/or xarray. We do want to make sure the S3 bucket and the Jupyter Hub are in the same region. 

I keep asking Shiv if there is some way of treating the UW Atmos TRMM ftp server like a virtual file system and he keeps patiently explaining NO so that's why I am suggesting working from a cloud copy. We can share the results with the *atmos* folks in case they are interested in our approach to data access. 

Shiv has the copy process running with only one login per directory; so it seems to be reasonably efficient and quick "per region-month". He can comment on wall clock time. If we take that to be 3 minutes then we have a total copy time of 3/60 * 20 * 17 * 12 = 220 hours or ten days. That's rather a lot of time but it is free to shove it into AWS. 

@Shiv this would be a good time to confirm / fix my numbers.


