import httpx
from fastapi import status

BASE_URL = "http://localhost:8000/corpuses"
JOBS_BASE_URL = 'http://localhost:8000/jobs'

### Create

# print("\n" + ("#" * 4) + "[POST corpuses]" + ("#" * 4) + "\n")

# data = {
#     "id" : "ls2n-duke",
#     "config" : {
#         "teams" : {
#             "hal" : ["LS2N-DUKE"],
#         },
#         "from_" : "2020-01-02T00:00:00Z",
#         "to" : "2026-01-02T00:00:00Z",
#         "page_size" : 32
#     }
# }


# response = httpx.post(BASE_URL, json=data)
# print(response.status_code)
# print(response.json())

# response = httpx.post(BASE_URL, json={ **data, "id" : 'to_delete' })
# print(response.status_code)
# print(response.json())

# assert response.status_code == status.HTTP_200_OK

# ### Get All

# print("\n" + ("#" * 4) + "[GET corpuses]" + ("#" * 4) + "\n")

# response = httpx.get(BASE_URL)
# print(response.status_code)
# print(response.json())

# assert response.status_code == status.HTTP_200_OK

### Launch Job

print("\n" + ("#" * 4) + "[POST /jobs/download]" + ("#" * 4) + "\n")

corpus_id = "ls2n-duke"
URL = f"{JOBS_BASE_URL}/download/{corpus_id}/start"

response = httpx.post(URL)
print(response.status_code)
print(response.json())

assert response.status_code == status.HTTP_200_OK

print("\n" + ("#" * 4) + "[SSE /jobs/download]" + ("#" * 4) + "\n")

job_id = response.json()['id']
URL = f"{JOBS_BASE_URL}/sse/{job_id}"

with httpx.stream("GET", URL, timeout=None) as response:
    
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            print(line)

exit()

### Delete

print("\n" + ("#" * 4) + "[DELETE corpuses]" + ("#" * 4) + "\n")

response = httpx.delete(f"{BASE_URL}/to_delete")
print(response.status_code)
print(response.json())

assert response.status_code == status.HTTP_200_OK

### Get All

print("\n" + ("#" * 4) + "[GET corpuses]" + ("#" * 4) + "\n")

response = httpx.get(BASE_URL)
print(response.status_code)
print(response.json())

assert response.status_code == status.HTTP_200_OK