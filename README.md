# Disaster Response Pipeline Project

### Installation
- Python (version 3.6.10)
- pandas `pip install pandas==1.0.3`
- numpy `pip install numpy==1.18.3`
- SQLAlchemy `pip install SQLAlchemy==1.3.18`
- nltk `pip install nltk==3.5`
- scikit-learn `pip install scikit-learn==0.22.2.post1` **(VERSION VERY IMPORTANT)**
- pipelinehelper `pip install pipelinehelper==0.7.6`
- plotly `pip install plotly==4.9.0`
- Flask `pip install flask==1.1.2`

### Project Motivation
This project aims to use the data provided by Figure Eight to set up a ELT pipeline and ML pipeline, to create a model and a web app.  
The model must classify disaster messages taking plain text as input  
The web app provides some basic insights into the model created as well as the data provided as well as serve as an interface for inputting text and obtaining the corresponding categories 

### File Descriptions
- The data/---.db file contains the cleaned DataFrame. This is the output of the ETL pipeline. It is accessed by the ML pipeline as well as the webapp.  
- The data/categories.csv and data/messages.csv file are the files provided by Figure Eight. These are given as input to the ETL Pipeline.  
- The data/process_data.py contains the ETL pipline which takes the input files as mentioned above, cleans, merges and stores them in a database.  
- The models/train_classifier.py file contains the ML pipeline, which takes the database object as input and uses Grid Search to produce a model for classification.  
- The app/run.py file contains the python script to run the server and send plotly graphs to the app/templates/master.html file.  
- The app/templates.master.html file acts as the homepage for the webapp. It shows visualisations which provide insight into the data.  
- The app/templates/go.html file, takes text input and uses the ML model to classify the message and displays the result.  

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse1.db`
    - To run ML pipeline that trains classifier and saves it 
        `python models/train_classifier.py data/DisasterResponse1.db models/best_model1.pkl`

    - Note - The ML Pipeline will take approximately 50 minutes to run...
    In case you want to test the application immediately, the pkl file is hosted on Google Drive (could not be hosted on github because the file is too large), and can be downloaded using this [link](https://drive.google.com/file/d/1U1wk65sQaLTUuBWcffWBhcUgEkgWBRI5/view?usp=sharing).   
    **Upon downloading, do not forget to place it in the models folder**


2. To run the web app, perform the steps given below
	The run.py file expects the model in the models folder to be named best_model1.pkl.  
	Therefore, before running the web app, DO ANY 1 OF THE BELOW TWO STEPS
	- Rename the model in the models folder to best_model1.pkl (if it is not already that)  
	- Change Line 43. in the run.py application to point to your model file  
	In the terminal, run:  
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Summary
- The model produced has an overall accuracy of 92% in identifying the correct category of message
- The Webapp provides 3 visualisations:  
	- Distribution of message Genres.  
	- Distribution of message Categories.  
	- Accuracy percentage for each Category obtained by model.  

### Acknowledgements
The data for this project was obtained from Figure Eight with the help of Udacity, as a part of a Nano-Degree Program
