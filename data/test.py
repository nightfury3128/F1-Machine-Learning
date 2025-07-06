import requests

url = 'https://api.openf1.org/v1/weather'
params = {

}
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    print(data)