## Polyglot Prompting: Multilingual Multitask Prompt Training
[**Overview**](https://hub.fastgit.org/neulab/SpanNER#overview) | 
[**Installation**](https://hub.fastgit.org/neulab/SpanNER#how-to-run) |
[**Polyglot Prompt Templates**](https://hub.fastgit.org/neulab/SpanNER#quick-installation) |
[**PolyPrompt Datasets**](https://hub.fastgit.org/neulab/SpanNER#demo) | 
[**Prepare Models**](https://hub.fastgit.org/neulab/SpanNER#prepare-models) |
[**Running**](https://hub.fastgit.org/neulab/SpanNER#how-to-run) |
[**System Combination**](https://hub.fastgit.org/neulab/SpanNER#system-combination) |
[**Bib**](https://hub.fastgit.org/neulab/SpanNER#bib)


This repository contains the code and datasets for our paper [Polyglot Prompting: Multilingual Multitask Prompt Training](https://arxiv.org/pdf/2204.14264.pdf).

## Overview
This paper aims for a potential architectural improvement for multilingual learning and asks: `Can different tasks from different languages be modeled in a monolithic framework, i.e. without any task/language-specific module? `

We approach this goal by developing a learning framework named Polyglot Prompting to exploit prompting methods for learning a unified semantic space for different languages and tasks with multilingual prompt engineering.
We performed a comprehensive evaluation of $6$ tasks, namely topic classification, sentiment classification, named entity recognition, question answering, natural language inference, and summarization, covering $24$ datasets and $49$ languages.  

<div  align="center">
 <img src="pic/polyprompt_frame.png" width = "700" alt="d" align=center />
</div>


## Polyglot Prompt Templates
- `./templates/CL` is the cross-languge prompt templates explored in this work.
- `./templates/IL` is the in-languge prompt templates explored in this work.


## Quick Installation

- `Python==3.7`
- `torch==1.9.0`
- `transformers==4.15.0`

Run the following script to install the dependencies,
```
pip3 install -r requirements.txt
```


## PolyPrompt Datasets

How to use the PolyPrompt Datasets?
We have released all the datasets prompted with the best settings. We provide two methods for downloading datasets.

### 1. Load the PolyPrompt Datasets from `DataLab`.

(1) Install `DataLab` with the following command:

```
pip install --upgrade pip
pip install datalabs
python -m nltk.downloader omw-1.4 # to support more feature calculation
```

More detailed instructions on installing `DataLab` can be found [here](https://github.com/ExpressAI/DataLab).


(2) After installing `DataLab`, the following code can be used to download/load datasets equipped with cross-language prompts.


```python
# pip install datalabs
from datalabs import load_dataset
dataset = load_dataset("poly_prompt","xquad.es")

# Get more information about the dataset.
print('dataset: ',dataset)
print(dataset['train'][0])
print(dataset['train']._info)
```
 
### 2. Build the PolyPrompt Datasets with our provided preprocessing code.

- `data_preprocess.py` is the data preprocessing code for seven target datasets (e.g., XNLI, TydiQA) and 15 non-target datasets (e.g., MCtest). One can use the prompt template to build the PolyPrompt Datasets.

- We also release the processed PolyPrompt Datasets. 
-  `7` target datasets with the cross-language prompt: XXX
-  `15` non-target datasets with the cross-language prompt: XXX
- The training set for the PolyPrompt model: XXX

The `PolyPrompt` model can be downloaded from XXX.













