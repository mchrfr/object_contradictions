# Object Contradictions
Author: Marie Freschlad

This repository contains the code used for the structural contradiction method described in the paper "Improving Language Model Performance by Training on Prototypical Contradictions" (submitted to the EUROPEAN CONFERENCE ON INFORMATION RETRIEVAL  - ECIR).

## Install the repository
1. Create a new virtual environment (required: Python >= 3.10):
```bash
$ uv venv object_contradictions_env --python 3.11
or
$ conda create -n object_contradictions_env python==3.11
```

2. activate your new environment:
```bash
$ source object_contradictions_env/bin/activate
or
$ conda activate object_contradictions_env
```

3. Navigate to the repository folder:
```bash
$ cd object_contradictions
```

4. Check whether you are in the right repository:
```bash
$ git status
```

5. Install the package:
```bash
$ uv pip install -e .
or 
$ pip install -e .
```

6. Install the spaCy transformer model:
```bash
$ uv pip install pip
$ uv run spacy download en_core_web_trf
or 
$ python -m spacy download en_core_web_trf
```

7. Select your .gitignore file and untrack your OpenAI access key. Make sure your .gitignore file includes the following line uncommented (without '#') and save your changes:
```bash
$ utils/openAI_key.py
```

8. After setting it as untracked, select your utils/openAI_key.py file and provide your OpenAI access key as the return value
```bash
$ ...
$ return 'ENTER YOUR API KEY HERE'
```
## Reproduce the experiments
Navigate to the /scripts directory.

Create structural contradictions:
    - create structural contradictions by running `script.py`;
    - find your created data set under /...;


### Important: You need to add your personal OpenAI API key under ./object_contradictions/utils/openAI_key.py, in order to make requests to the OpenAI API. Afterwards uncomment the utils/openAI_key.py - script in your .gitignore file