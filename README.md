# Cadiovascular-prediction
Cadiovascular Prediction with ANN 


1-INTRODUCTION
Cardiovascular diseases (CVDs) are a group of disorders of the heart and blood vessels, according to the world health organization, CVDs are the number 1 cause of death globally: more people die annually from CVDs than from any other cause, an estimated 17.9 million people died from CVDs in 2016, representing 31% of all global deaths. Of these deaths, 85% are due to heart attack and stroke [1]. People with cardiovascular disease or who are at high cardiovascular risk need early detection and management using counseling and medicines, as appropriate, Heart attacks and strokes are usually acute events and are mainly caused by a blockage that prevents blood from flowing to the heart or brain.
The most important behavioral risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use, and harmful use of alcohol. The effects of behavioral risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. These “intermediate-risk factors” [1] can be measured in primary care facilities and indicate an increased risk of developing a heart attack, stroke, heart failure, and other complications. 
The usage of Machine Learning and deep learning in healthcare increase as the amount and the complexity of medical data increase that means the more data the more needs for deep learning which are a computational approach based on a large collection of neural units loosely modeling the way the brain solves problems,
 Machine Learning Ensemble Methods and Artificial Neural Networks are used to build prediction system that help in the primary prediction of cardiovascular abnormalities based on initial medical examination such as Glucose level, Cholesterol level, Systolic blood pressure, Diastolic blood pressure, and patient health information such as weight, BMI, Alcohol intake, Physical Activity and Smoking 



2- DATA SET

The data contains 70,000 observations on Three types of input features: objective features which is a piece of factual information (age, height, weight, gender), examination features which is a result of medical examination as (systolic blood pressure, diastolic blood pressure, cholesterol, glucose, physical activity), and Subjective features that represent information given by the patient as (smoking, alcohol intake). and the target is a binary variable that indicates the presence or absence of cardiovascular disease, we split the dataset into a training set with 80% of the data and test the model with 20% of the data, data available as open source [2]
3- METHODOLOGY


3.1 Data preprocessing and preparation 

In the stage of prepare data for machine learning workflow. It’s needed to transform the data in a way that a computer would be able to work with it and choose suitable data types that achieve good performance in computational processes, exclude noise that may damage the accuracy and the efficiency of the model

3.1.1 - Find the correlation between cardiovascular disease and each Factor

In order to select which features can make a high correlate with the target variable, the height of the patient may not affect cardiovascular disorders occurrences, we can use the height and the weight of the patient to get the body mass index which is correlated with the target variable by 0.165, Plotting heatmap of the dataset shown in Fig 1. gives us the intuition about the features that have low correlation with cardiovascular disorders occurrences, from the heatmap we observe that gender with 0.008 and alcohol intake with -0.007 have very low correlation ratio so they can be excluded from the training process 

![image](https://user-images.githubusercontent.com/83831812/117480297-b6e3c100-af61-11eb-8977-83ac4829ae86.png)


3.1.2 - Remove the outliers

An outlier is an unusually large or small observation. Outliers can have a disproportionate effect on statistical results, which can result in misleading interpretations. In our dataset, we detect outliers through box plot which indicate a huge outlier in systolic blood pressure measures which goes up to 16000 while the interquartile range indicates the upper limit is about 170 mmHg and the lower limit is about 90 mmHg, and outliers in diastolic blood pressure go up to 10000 mmHg while the interquartile range indicates the upper limit is about 105 and the lower limit is about 65 mmHg, by removing the outlier in systolic blood pressure, diastolic blood pressure, and BMI the dataset is ready for the standardization step.

![image](https://user-images.githubusercontent.com/83831812/117480417-dd096100-af61-11eb-93b6-1427012aabf3.png)



3.1.3 - Data standardization

Data standardization is a scaling technique where the values are centered around the mean with a unit standard deviation. it is a critical process of bringing data into a common format that allows for collaborative research, large-scale analytics, and sharing of sophisticated tools and methodologies [3], Healthcare data can vary greatly from one organization to the next. Data are collected for different purposes, such as clinical research, and direct patient care. These data may be stored in different formats using different database systems and information models, and despite the growing use of standard terminologies in healthcare, the same concept (e.g., blood glucose) may be represented in a variety of ways from one setting to the next. standardization criterion is given by

Xstand=  (X-mean(X))/(standard deviation(X))






3.2 Machine Learning Model 

The battle of getting a satisfying accuracy in healthcare is very sensitive, working with some of traditional machine learning algorithm may not be enough in some cases where data haven’t a specific pattern, so we try to work on artificial neural networks and compare its results with ensemble methods and improve the accuracy with semi-supervised learning to build machine learning system that can predict cardiovascular disorders.

3.2.1 – Artificial Neural Networks  

An Artificial Neural Network (ANN) is the component of artificial intelligence that is meant to simulate the functioning of a human brain in analyzes and processes information and solve the problem [4], general steps [5] of the neural network is as follows: a) Network structure is defined with a fixed number of nodes in input, output, and hidden layer. Shown in Fig2, b) An algorithm is used for the learning process, our ANN model consists of the Input layer which represents the features, 3 Hidden layers with 32,16, and 6 nodes respectively, and an Output layer which represents the desired target




             
![image](https://user-images.githubusercontent.com/83831812/117480612-1f32a280-af62-11eb-906d-32ed41b34e7b.png)











3.2.2 – Ensemble Method with Random Forest   

Random Forest developed by Leo Breiman [6] is a group of un-pruned classification or regression trees made from the random selection of samples of the training data. Random features are selected in the induction process. Prediction is made by aggregating (majority vote for classification or averaging for regression) the predictions, Random Forest generally exhibits a significant performance improvement as compared to single tree classifier as it combines the output of multiple (randomly created) Decision Trees to generate the final output, the generalization error rate that it yields compares favorably to Adaboost, however it is more robust to noise. Using GridSearch technique with the estimator of RandomForestClassifier will determine the best hyperparameters for the training process.





![image](https://user-images.githubusercontent.com/83831812/117480667-2fe31880-af62-11eb-925b-065517eab7b2.png)






	












3.2.3 – Semi-Supervised Learning with Logistic Regression    

Semi-supervised learning is the type of machine learning that uses a combination of a small amount of labeled data and a large amount of unlabeled data to aids and bias the clustering of unlabeled data [7]. This approach of machine learning is a combination of supervised machine learning, which uses labeled training data, and unsupervised learning, which uses unlabeled data, semi-supervised use of unlabeled data in conjunction with a small amount of labeled data can produce considerable improvement in learning accuracy [8]. A semi-supervised clustering technique is used to label the unlabeled data and further helps to improve the training of the classifier. Shown in Fig 5, The logistic regression classification is used for the process of classification to provide a better internal structure of the data in the process of semi-supervised clustering. Both labeled and unlabeled data are used by the model and help to classify the unlabeled data with better results, the simplicity of the classification process helps to give better results and is more efficient [8].


![image](https://user-images.githubusercontent.com/83831812/117480712-40938e80-af62-11eb-8cf4-6871f7887960.png)



4- EXPREMENTL RESULT

Our experiments were conducted on Python 3.7 with 70,000 observations, TensorFlow with Keras framework for ANN model, Sci-kit Learn framework for Random Forest model and Logistic regression with Pytorch framework for semi-supervised learning, we get training and testing accuracy respectively, 72,81% and 73.83 for ANN model, 87.06% and 70.01% for random forest algorithm and 97.8% of both training and testing accuracy with semi-supervised model. 

 




![image](https://user-images.githubusercontent.com/83831812/117480827-615be400-af62-11eb-8bed-be9f602d088f.png)






5- DISCUSSION 


Early diagnosis of diseases plays a vital role in the treatment process. The diagnosis of Cardiovascular disease at an early stage can reduce the damage to the heart and prevents heart failure,  building a prediction system for heart abnormalities is a great way to prevent heart disease provide the system with the basic information of the patient and some essential  measures such as glucose level and blood pressure will result in predict whether there is a cardiovascular disorders or not in early stage, from the comparison of the  accuracies we get from the three algorithm that we have applied on the data, we conclude that the semi-supervised technique has a high impact on increasing the accuracy of our prediction system of cardiovascular disorder ,as the 73% accuracy of ANN model make it need for more training data to and the huge difference between the training accuracy and testing accuracy of random forest indicate that there is a big overfitting and this make to conclude that that the traditional classifiers with a very similar data points is not a good choose .

8- REFERENCES

	https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

	https://www.kaggle.com/sulianova/cardiovascular-disease-dataset

	Gal, Michal and Rubinfeld, Daniel L., Data Standardization (June 2019). 94 NYU Law Review (2019) Forthcoming, NYU Law and Economics Research Paper No. 19-17

	A. Basheer and M. Hajmeer, “Artificial neural networks: fundamentals, computing, design, and application,” J. Microbiol. Methods, vol. 43, no. 1, pp. 3–31, 2000.

	Z. Dokur and T. Ölmez, “ECG beat classification by a novel hybrid neural network,” Comput. Methods Programs Biomed., vol. 66, no. 2, pp. 167–181, 2001

	Breiman, L., Random Forests, Machine Learning 45(1), 5-32, 2001.

	- Blum, A. and Mitchell, T. “Combining Labeled and Unlabeled Data with Co-Training” Proceedings in 11th Annual Conf. on Computational Learning Theory ,1998

	Blum, A. and Chawla, S. Learning from label and unlabeled data using graph mincuts, Proceedings in 18th International Conference in Machine Learning, 2001

	 Pachghare, V. Khatavkar, V. “Pattern Based Network Security Using Semi-Supervised Learning” in   International Journal of Information and Network Security (IJINS) , 2012





 


