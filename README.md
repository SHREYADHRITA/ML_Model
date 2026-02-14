# ML_Model
This repository contains various models to analyse a linear classification model 
and predict the output using that model


Problem Statement-
---------------------------------------------------------------------------------
We need to implement multiple classification models. Build an interactive 
Streamlit web application to demonstrate your models and then deploy the 
app on Streamlit Community Cloud

Dataset used for creating this model- 
https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset

The Early Stage Diabetes Risk Prediction model helps identify whether an individual is 
at risk of developing diabetes. By analyzing relevant health data, the model enables 
early detection of potential diabetic conditions, allowing timely treatment. This proactive 
approach can significantly improve health outcomes and reduce long-term complications.

ML Model Name	        Accuracy Precision 	Recall 	F1 Score    MCC	    AUC
logistic Regression 	0.9423	 0.946	    0.9423	0.9428	    0.8832	0.9902
Decision Tree	        0.8942	 0.8949	    0.8942	0.8945	    0.7778	0.9277
kNN	                    0.9327	 0.9383	    0.9327	0.9333	    0.8653	0.9697
Naive Bayes 	        0.9423	 0.9435	    0.9423	0.9426	    0.88	0.9863
Random Forest(Ensemble) 0.9904	 0.9906	    0.9904	0.9904	    0.98	1
XGBoost (Ensemble) 	    0.9712	 0.9715	    0.9712	0.9712	    0.9395	0.9885
