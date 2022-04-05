from re import A
import requests
import pandas as pd
import json

def news():
    url = ('https://gnews.io/api/v4/search?q=agriculture india&token=36faec3e6c6a2a656d59b587a8084a5e')
    response = requests.get(url)
    print(response)
    a = response.json()
    c = pd.DataFrame(a['articles'])
    d = c.head(6)
    print(d)
    return d.title


