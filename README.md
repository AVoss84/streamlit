# Code generator using OpenAI's GPT-3 

## Installation

The app was tested with Python 3.7

### Requirements

To install your application locally, execute the following steps:

```bash
${CONDA_PATH}/conda env create -f environment.yml
```

Activate your environment:

```bash
$ conda activate streaml
```

Install other packages using pip:

```bash                                 
$ pip install -r requirements.txt             
```

Start your application by running:

```bash                                 
$ streamlit run app.py
```

Note: You might also want to further customize your app by creating a 'config.toml' file in your '~/.streamlit' folder, e.g. for port modifications or any custom app settings
