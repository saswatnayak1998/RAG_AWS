import urllib.request
import ssl

context = ssl._create_unverified_context()
urllib.request.urlopen("https://docs.aws.amazon.com/", context=context)

import requests

response = requests.get("https://docs.aws.amazon.com/", verify=False)