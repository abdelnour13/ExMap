import httpx
from fastapi import status
from src.envirenment import env

BASE_URL = "http://localhost:8000/kgs"
JOBS_BASE_URL = 'http://localhost:8000/jobs'

### Create

print("\n" + ("#" * 4) + "[POST Knowledge Graph Construction]" + ("#" * 4) + "\n")

data = {
    "id" : "ls2n-duke",
    "config" : {
        "corpus_id" : "ls2n-duke",
        "llm_name" : "deepseek-flash"
    }
}


response = httpx.post(BASE_URL, json=data)
print(response.status_code)
print(response.json())

response = httpx.post(BASE_URL, json={ **data, 'id' : 'to_delete' })
print(response.status_code)
print(response.json())

### Launch Job

print("\n" + ("#" * 4) + "[POST /jobs/kgs]" + ("#" * 4) + "\n")

corpus_id = data['config']['corpus_id']
URL = f"{JOBS_BASE_URL}/kg_construction/{corpus_id}/start"

response = httpx.post(URL, headers={ "X-Api-Key" : env.get_api_key('deepseek') })
print(response.status_code)
print(response.json())

assert response.status_code == status.HTTP_200_OK

print("\n" + ("#" * 4) + "[SSE /jobs/kgs]" + ("#" * 4) + "\n")

job_id = response.json()['id']
URL = f"{JOBS_BASE_URL}/sse/{job_id}"

with httpx.stream("GET", URL, timeout=None) as response:
    
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            print(line)

### Delete

print("\n" + ("#" * 4) + "[DELETE kgs]" + ("#" * 4) + "\n")

response = httpx.delete(f"{BASE_URL}/to_delete")
print(response.status_code)
print(response.json())

assert response.status_code == status.HTTP_200_OK

### Get All

print("\n" + ("#" * 4) + "[GET KGs]" + ("#" * 4) + "\n")

response = httpx.get(BASE_URL)
print(response.status_code)
print(response.json())

assert response.status_code == status.HTTP_200_OK