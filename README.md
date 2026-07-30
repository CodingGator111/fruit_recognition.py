Fruit Recognition and Identification

 

 This project is designed for classifying and identifying different images of fruit and other grocery items, and then giving a concise description about what they are, where they're grown, and more (tailored to the user's response). Responses by the computer are given by Ollama. This model was trained on the following dataset containing over 180,000 images on different kinds of fruits. This dataset is from Kaggle: https://www.kaggle.com/datasets/moltean/fruits/data
 
 
 
 
 To initiate the response, you can run the docker container with:
  sudo docker run -it --rm --runtime nvidia --network host \
  -v ~/fruit_recognition:/workspace/fruit_recognition \
  -v ~/fruit_dataset_split:/home/nvidia/fruit_dataset_split \
  dustynv/l4t-pytorch:r36.2.0

  Then install zstd and ollama with:

  apt-get update && apt-get install -y zstd
  curl -fsSL https://ollama.com/install.sh | sh

  
![add image descrition here](direct image link here)
![The output of the code](image-1.png)

## The Algorithm

This project relies on PyTorch and torchvision to handle operations involving deep learning, along with allocating GPU memory via CUDA, dataset transforming, and executing the ResNet-18 neural network. Ollama is an LLM that is an inference engine running a light language model on the Orin's GPU. 


## Running this project

-> Before running this project, make sure to install torch, torchvision, pillow (PIL), and ollama (install zstd first before ollama)
1. start up ollama in the background with: ollama serve &
2. Pull the LLM model: ollama pull llama3.2:1b and attach your question about a fruit:

Sample question: ollama pull llama3.2:1b "Where do dragonfruits grow best?"

The program should then output a response.

If you want to hide the background processes (to make the output cleaner), you can put the command: ollama serve > /dev/null 2>&1 &
-> Then run ollama run llama3.2:1b "(your question)"

[View a video explanation here](video link)
