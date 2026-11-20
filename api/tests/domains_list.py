import httpx
from fastapi import status
from src.envirenment import env

print("#### GET domains_list ####")

BASE_URL = "http://localhost:8000/domains_list"

response = httpx.get(BASE_URL)
print(response.status_code)
print(response.json())

print("#### POST domains_list ####")

data = {
    "name" : 'test',
    "domains" : [
        {
            "sub_domain" : "computer science",
            "domain" : "computer science",
            "definition" : "computer science"
        }
    ]
}

response = httpx.post(BASE_URL, json=data)
print(response.status_code)
print(response.json())