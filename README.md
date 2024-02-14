This is a question classification model that classifies questions into 4 classes according to the grammatical context of the questions. 

How to Run Question Classification Model

Step 1: Download the following model and the script file from our github link: 
https://github.com/USC-Lyon-Lab/question_classification
Model file name - question_classfication
Script file name - predictor.ipynb

Step 2: Make a new folder on your computer and put both of the above files in this folder. 
Also, place the csv file that needs to be classified in the same folder. 

Step 3: Make sure there is a column named “questions” in the csv file that needs to be classified. Please note, the column text is case sensitive and there should not be any spaces before or after the column name.

Step 4: Open visual studio and click on ‘File’->‘Open Folder’ and browse to the above folder. 
Once opened, we can see all the content of the folder on the left side in Visual Studio.

Step 5: Open the predictor.py file in visual studio by clicking on the file name shown on the left window. 
In the second cell of the predictory.py file, there is a variable called “file_path”. Update the value of this variable to your csv file name. Eg: if your csv file name is “roberta.csv” then update the value: -
file_path = “./roberta.csv”

Step 6: Click on the “Run All” button on top to run the script. 

Step 7: Once the script finishes, the csv file will be updated with a new column called ‘predictions’.

