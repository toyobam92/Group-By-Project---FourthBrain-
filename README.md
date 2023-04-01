# Group By Project - Customizable Uplift Model

This repository contains the source code for a customizable uplift model platform developed as part of the Group By Project at FourthBrain. The purpose of this project is to provide an easy-to-use framework for uplift modeling, which can be applied to various business use cases such as targeted marketing, customer retention, and revenue optimization.

## Table of Contents
### Introduction
### Installation
### Usage
### Customization
### Results and Evaluation
### Demo
### Contributing
### License

## Introduction
Uplift modeling is a sub-field of machine learning that focuses on estimating the causal impact of a treatment (such as a marketing campaign) on a specific outcome (such as customer conversion). This model aims to predict the difference in the outcome between the treated and control groups, helping businesses make informed decisions about which customers to target with their campaigns.

## Installation
To get started, clone the repository to your local machine using the following command:

git clone https://github.com/toyobam92/Group-By-Project---FourthBrain-/tree/final/.git
After cloning the repository, navigate to the project directory and install the required dependencies using the following command:

pip install -r requirements.txt

## Usage
To run the customizable uplift model on your data, follow these steps:

Prepare your dataset in CSV format with the following columns:

treatment_tag: A binary variable indicating whether a customer received the treatment (1) or not (0).
conversion: A binary variable indicating the target outcome, such as customer conversion (1) or not (0).
Additional features that may influence the outcome (e.g., demographics, purchase history).

Run the experimentation notebook Uplift_model.ipynb on Jupyter using the MLRun framework. This notebook will train the model on the data and save the trained model in the mlrun folder. The model will then be moved to an S3 bucket for storage and deployment purposes. The training and testing datasets will be saved in CSV format under the dat folder.

Deploy the main.py script on an AWS instance to run the Streamlit app using streamlit run main.py, which provides an interactive interface to for the campaign and model results 

## Customization
You can customize the uplift model by modifying the settings in the Uplift_model.ipynb notebook, such as:

Choosing different uplift modeling techniques, such as Two-Model, Solo-Model and Class Transformation model and specifying the underlying classifier (e.g., logistic regression, random forest,XGBoost).

Defining the performance metrics used to evaluate the model, such as area under the uplift curve (AUUC), Qini coefficient, and response rate.

## Results and Evaluation
The model will output various performance metrics, including AUUC, Qini coefficient, and response rate, to help you assess the effectiveness of the uplift model. Additionally, the results will be visualized using charts to facilitate interpretation.

## Demo
Visit the live demo of the Streamlit app deployed on AWS: http://3.235.28.171:8501/. The app allows  visualize the results interactively the campaign results and model results interactively 

## Team Members
Our team consists of the following members:

Toyosi Bamidele

Uchenna Mgbaja

## Contributing
We welcome contributions to improve and extend the functionality of this project. If you're interested in contributing, please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
