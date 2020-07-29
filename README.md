# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse1.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse1.db models/best_model1.pkl`

Note - The ML Pipeline will take approximately 50 minutes to run...
In case you want to test the application immediately, the pkl file is hosted on Google Drive (could not be hosted on github because the file is too large), and can be downloaded using this [link](https://drive.google.com/file/d/1U1wk65sQaLTUuBWcffWBhcUgEkgWBRI5/view?usp=sharing)
**Upon downloading, do not forget to place it in the models folder**

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
