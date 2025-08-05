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
