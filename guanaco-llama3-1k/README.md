---
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 1777268
    num_examples: 1000
  download_size: 976999
  dataset_size: 1777268
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

Guanaco-1k for Llama3 

I took a page out of mlabonne's book and made a subset of the original [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
dataset that matches the Llama3 formatting. I used the same [colab notebook](https://colab.research.google.com/drive/1Ad7a9zMmkxuXTOh1Z7-rNSICA4dybpM2?usp=sharing#scrollTo=C50UUxdE_d14) as the original. Here's the changed code 

```
from datasets import load_dataset
import re

# Load the dataset
dataset = load_dataset('timdettmers/openassistant-guanaco')

# Shuffle the dataset and slice it
dataset = dataset['train'].shuffle(seed=42).select(range(1000))

# Define a function to transform the data
def transform_conversation(example):
    conversation_text = example['text']
    segments = conversation_text.split('###')

    reformatted_segments = []

    # Iterate over pairs of segments
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace('Human:', '').strip()

        # Check if there is a corresponding assistant segment before processing
        if i + 1 < len(segments):
            assistant_text = segments[i+1].strip().replace('Assistant:', '').strip()

            # Apply the new template
            reformatted_segments.append(f'<|start_header_id|>user<|end_header_id|>{{{{{human_text}}}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{{{{{assistant_text}}}}}<|eot_id|>')
        else:
            # Handle the case where there is no corresponding assistant segment
            reformatted_segments.append(f'<|start_header_id|>user<|end_header_id|>{{{{{human_text}}}}}<|eot_id|>')

    return {'text': ''.join(reformatted_segments)}


# Apply the transformation
transformed_dataset = dataset.map(transform_conversation)
```