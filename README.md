# Streamlit dashboard

## Installation

The app was tested with python 3.7

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

You might want to customize your app settings by creating 'config.toml' file in your '~/.streamlit' folder, e.g. for port settings or any custom app settings
