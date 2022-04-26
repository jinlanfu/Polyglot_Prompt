# Polyglot_Prompt


## How to use the PolyPrompt Datasets?

```python
# pip install datalabs
from datalabs import load_dataset
dataset = load_dataset("poly_prompt","xquad.es")

# Get more information about the dataset.
print('dataset: ',dataset)
print(dataset['train'][0])
print(dataset['train']._info)
```
 

