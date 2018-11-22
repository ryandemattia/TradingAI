import requests
def get_endpoint(self, ep):
    r = requests.get(url='')
    print(r.json())