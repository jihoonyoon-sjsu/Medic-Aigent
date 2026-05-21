# CMPE 256 — Final Project

Medication recommendation/ranking. We use the MIMIC-IV data set.

Everything runs from 'final_notebook.ipynb' in Google Colab.

## Setup

1. Open the notebook in colab. You can also open in jupyter, but change file paths in the notebook accordingly.
2. Set PROJECT_DIR to the folder where your notebook is located
3. Download the data folder from here:
https://drive.google.com/file/d/1FCH_-mDz7NBdiulAeONB7mrdG7RH0KkJ/view?usp=share_link
4. Put that folder in your Google drive, where your notebook is. 
5. Pick a runtime. Some models only run on CPU, some utilize GPU.
6. Run it. Necessary packages should download as you run the cells.

## Running the front end:
cd into the /app folder
run these commands in order:
1. python3 -m venv .venv
2. source .venv/bin/activate (macOS) (.venv\Scripts\activate for Windows)
3. pip install -r requirements.txt
4. streamlit run app.py

## Note about requirements.txt
Requirements.txt in this notebook is just for the front end. It is located in the /app subfolder. All packages required for the Colab notebook are included in the Colab environment or installed in the cells.
