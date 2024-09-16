# Behavior tracking using VLM

## Description
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. By using pretrained model CLIP from OpenAI, I plan to build a behavior tracking model to detect people behavior. This project is still in demo.

## Setup
First, install: 

```pip install requirements.txt```

Then, install CLIP:

```pip install git+https://github.com/openai/CLIP.git```

To run model:

```python behavior_vlm.py```


