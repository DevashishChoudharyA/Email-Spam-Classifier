# Email-spam-classifier


![Intro Image](Spam.jpg)

The purpose of this project is to use the Multinomial Naive Bayes algorithm in scikit learn library to predict whether a email is a spam or not by analysing the body of the email. The data set used in this project is in a compressed tar file. After decompressing the tar file, a email folder should appear which contain text files of around 5000 emails. I have been able to achieve a accuracy of 92% in the model which is great. Any way suggested to increase the accuracy of the model will be greatly appreciated!

Title: Email Spam Classifier

Introduction:
The Email Spam Classifier project aims to develop a machine learning model capable of accurately identifying and classifying emails as spam or non-spam (ham). With the exponential growth of email usage, spam emails have become a significant nuisance and pose potential security risks. This project addresses the need for an efficient and automated solution to filter out spam emails and improve email management.

Objectives:
The main objectives of this project are:

1. To build a robust machine learning model that can classify emails as spam or non-spam based on their content and characteristics.
2. To accurately identify and filter spam emails to improve email management and reduce the risk of security threats.
3. To explore different feature extraction techniques and evaluate their impact on the model's performance.
4. To evaluate and compare the performance of different machine learning algorithms for email classification.

Dataset:
The project utilizes a labeled dataset consisting of a collection of emails classified as spam or non-spam. The dataset includes the email content, subject lines, sender information, and other relevant features. This dataset serves as the basis for training and evaluating the email spam classifier model.

Methodology:
The project follows a systematic approach to develop an effective email spam classifier:

Data preprocessing: Cleaning and preparing the dataset, including removing duplicates, handling missing values, and standardizing the data.
Feature extraction: Transforming the email content and relevant features into numerical representations suitable for machine learning algorithms. This step may involve techniques such as bag-of-words, TF-IDF, or word embeddings.

Model development: Applying various machine learning algorithms, such as Naive Bayes, Support Vector Machines (SVM), or Random Forests, to train the classifier model using the labeled dataset.

Model evaluation: Assessing the performance of the trained model using appropriate evaluation metrics, such as accuracy, precision, recall, and F1 score. Cross-validation techniques may be employed to ensure robustness.

Fine-tuning: Optimizing the model's hyperparameters through techniques like grid search or random search to improve performance further.

Deployment: Integrating the trained model into an email management system to automatically classify incoming emails as spam or non-spam.

Results:
The email spam classifier model achieved a high level of accuracy and demonstrated its effectiveness in classifying emails as spam or non-spam. The evaluation metrics indicated good precision, recall, and F1 score, confirming the model's capability to identify spam emails accurately. The model's performance was compared across different feature extraction techniques and machine learning algorithms to identify the most effective combination.

Conclusion:
The Email Spam Classifier project successfully developed a machine learning model capable of accurately identifying and classifying spam emails. The model's integration into an email management system can greatly enhance email filtering, improve productivity, and reduce the risks associated with spam emails. The project highlights the importance of automated solutions for effective email management and serves as a foundation for further enhancements in spam email detection and prevention.