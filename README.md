# Project Digital Twin
Tracking various metric using satellite imagery

## Features

- Connects to satellite APIs
- Processes satellite data into useful information
- Provides visualizations and analysis tools

## Setup Instructions

1. Clone this repository and navigate to the project folder. Then create the Conda environment with the following command:

`conda env create -f environment.yml`

Activate the environment with

`conda activate satellite_nso`

All the packages and dependencies will have been installed automatically.

2. Create a `.env` file in the root directory of the project with your own credentials:

`API_USERNAME=your_username`  
`API_PASSWORD=your_password`


3. **Note:** The `.env` file is ignored by Git, so your credentials won’t be accidentally pushed to GitHub.

4. Run the project as usual. The credentials will be securely loaded from the `.env` file.

## Steps for replacing .ipynb for .py

1. Switch to your feature branch with

`git checkout your-branch-name`

2. Pull the latest change from main:

`git pull origin main`

After this there should be a .py file in src/

3. Stop tracking the .ipynb file using

`git rm --cached src/Project.ipynb`

4. Commit the change

If you want to convert your .ipynb file to .py, you can use jupytext:

`pip install jupytext`
`jupytext src/Project.ipynb --to py`

Create a new cell in the Python file with `# %%`

## Setting up the webapp

To run the webapp, run the `web_backend.py` file.
