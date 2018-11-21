import requests
r = requests.get(url='https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty')
print(r.json())