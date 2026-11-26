### NVIDIA-TOOLKIT
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu18.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

sudo ln -s /sbin/ldconfig /sbin/ldconfig.real # YOU MAY NEED TO RUN THIS

### TEST NVIDIA-CONNECTIVITY (nvidia-drivers should be installed on the host machine)
sudo docker run --name=test --gpus all -it ubuntu
sudo docker exec -it test nvidia-smi