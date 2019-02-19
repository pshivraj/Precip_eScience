## TRMM Data Clustering multi instances

This repo consists of file `multiple_instance.py` which uses AMI's to spin up multiple clusters and run clustering script ```clusterEvents.py``` and terminates on completion.
## Usage

Before executing the `data_fetch.py` initial setup requires setting up requisite credential file for AWS S3.

Provide one time `key_id` and `key_access` to be stored in the `home` folder as `creds_multi.json` which will be used by `multiple_instance.py`.
```
import os
from os.path import expanduser
import json

home = expanduser("~")
# store one time credentials in the home directory
creds = {'key_id' : '',
         'key_access' : ''}
with open(os.path.join(home,'creds_multi.json'), 'a') as cred:
    json.dump(creds, cred)
```

Post setting up credentials next steps involves :

Class `Multi_instance()` needs to be initialized with:
  
      1.) Providing AMI id to spin instances: self.AMI = ""
      2.) Providing Security group for the AMI to attach to : self.SECURITY_GROUP = ['']
      
## Runtime

For `EPO` it takes ~12 hours on a c5.4x large machine scaled over 8 instances.
