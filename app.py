from flask import Flask, render_template, request
import pandas as pd
import joblib
from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from jinja2 import Template
from sklearn.metrics import f1_score
import numpy as np
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            data = pd.read_csv(file)

            with open('LR.pkl', 'rb') as f:
                model1 = pickle.load(f)
            
            with open('NB.pkl', 'rb') as f:
                model2 = pickle.load(f)
            
            with open('DT.pkl', 'rb') as f:
                model3 = pickle.load(f)

            X = data.drop('target', axis=1) 
            y_true = data['target']  

            y_pred1 = model1.predict(X)
            y_pred2 = model2.predict(X)
            y_pred3 = model3.predict(X)


            conf_matrix1 = confusion_matrix(y_true, y_pred1)
            f1score1 = f1_score(y_true, np.round(y_pred1))
            print('f1_score for Model1 test:',f1score1)
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix1, display_labels=model1.classes_)
            disp.plot(values_format='')  
            plt.title("Logistic Regression Confusion Matrix")
            plt.savefig('static/confusion_matrix_plot1.png')  

            conf_matrix2 = confusion_matrix(y_true, y_pred2)
            f1score2 = f1_score(y_true, np.round(y_pred2))
            print('f1_score for Model2 test:',f1score2)
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix2, display_labels=model2.classes_)
            disp.plot(values_format='') 
            plt.title("Naive Bayes Confusion Matrix")
            plt.savefig('static/confusion_matrix_plot2.png')


            conf_matrix3 = confusion_matrix(y_true, y_pred3)
            f1score3 = f1_score(y_true, np.round(y_pred3))
            print('f1_score for Model2 test:',f1score3)
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix3, display_labels=model3.classes_)
            disp.plot(values_format='')  
            plt.title("Decision Tree Confusion Matrix")
            plt.savefig('static/confusion_matrix_plot3.png')

            return render_template('result.html', f11=[f1score1,f1score2,f1score3], plot_filename1='confusion_matrix_plot1.png',plot_filename2='confusion_matrix_plot2.png',plot_filename3='confusion_matrix_plot3.png')
        else:
            return "No file provided"


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
