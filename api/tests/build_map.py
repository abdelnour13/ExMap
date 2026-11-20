import httpx
from fastapi import status
from src.envirenment import env

BASE_URL = "http://localhost:8000/maps"
JOBS_BASE_URL = 'http://localhost:8000/jobs'

### Create

print("\n" + ("#" * 4) + "[POST Graph Learning]" + ("#" * 4) + "\n")

data = {
    "id" : "ls2n-duke",
    "config" : {
        "model" : "ls2n-duke",
        "llm" : "deepseek-flash",
        "name" : "LS2N DUKE",
        "description" : "This is a test.",
        "clustering_alg" : "leiden",
        "resolution" : 15
    }
}


response = httpx.post(BASE_URL, json=data)
print(response.status_code)
print(response.json())

response = httpx.post(BASE_URL, json={ **data, 'id' : 'to_delete' })
print(response.status_code)
print(response.json())

### Launch Job

# print("\n" + ("#" * 4) + "[POST /jobs/expertise_maps]" + ("#" * 4) + "\n")

# corpus_id = data["id"]
# URL = f"{JOBS_BASE_URL}/expertise_maps/{corpus_id}/start"

# response = httpx.post(URL, headers={ "X-Api-Key" : env.get_api_key('deepseek') })
# print(response.status_code)
# print(response.json())

# assert response.status_code == status.HTTP_200_OK

# print("\n" + ("#" * 4) + "[SSE /jobs/expertise_maps]" + ("#" * 4) + "\n")

# job_id = response.json()['id']
# URL = f"{JOBS_BASE_URL}/sse/{job_id}"

# with httpx.stream("GET", URL, timeout=None) as response:
    
#     response.raise_for_status()

#     for line in response.iter_lines():
#         if line:
#             print(line)

### Delete

print("\n" + ("#" * 4) + "[DELETE /experiments]" + ("#" * 4) + "\n")

response = httpx.delete(f"{BASE_URL}/to_delete")
print(response.status_code)
print(response.json())

assert response.status_code == status.HTTP_200_OK

### Get All

print("\n" + ("#" * 4) + "[GET experiments]" + ("#" * 4) + "\n")

response = httpx.get(BASE_URL)
print(response.status_code)
print(response.json())

assert response.status_code == status.HTTP_200_OK