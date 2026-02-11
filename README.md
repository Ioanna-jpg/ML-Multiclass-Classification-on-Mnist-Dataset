# 🧠 Multiclass Classification – From Scratch Implementations

## 📌 Overview
This project focuses on implementing fundamental machine learning algorithms **from scratch** and applying them to a **multiclass classification** problem.

The purpose is not to achieve the best predictive performance but to understand, build, and experimentally evaluate the methods presented in lectures and tutorial sessions.

---

## 🎯 Requirements for the Dataset
The dataset used in this project must:

- contain **at least three (3) classes**
- contain **at least three (3) features**
- **NOT** be from the MNIST family (digits, fashion MNIST, etc.)

Different datasets may be used for different algorithms if necessary.

---

## 🧪 Implemented Methods

### 1. Principal Component Analysis (PCA)
Used for dimensionality reduction.

If the original data has **n** features, PCA must be applied to reduce the data to a dimension:
m ∈ [2, n)

It is sufficient to demonstrate PCA on the dataset.  
Using the transformed features in the other models is optional.

---

### 2. Least Squares (adapted for classification)
The least squares algorithm, originally introduced for regression, must be modified to operate in a classification setting.

---

### 3. Logistic Regression
The model must be trained using:

- **Stochastic Gradient Descent (SGD)**
- **Cross Entropy Loss**

Required outputs:
- accuracy on training and test sets  
- plot of cross entropy loss per epoch (train & test)

---

### 4. K-Nearest Neighbors (KNN)
The classifier must be evaluated for:
K = 1 to 10

Results for each value of **K** should be presented.

---

### 5. Naïve Bayes (Gaussian)
⚠️ Binary feature conversion (0/1) is **not allowed**.

Instead:
- assume Gaussian distributions  
- use diagonal covariance matrices  
- covariance matrices do not need to be shared across classes  

---

### 6. Multilayer Perceptron (MLP) – PyTorch
A feed-forward neural network with linear layers and nonlinear activations.

You are free to select:
- number of layers  
- activation functions  
- learning rate  
- any other hyperparameters  

Required outputs:
- accuracy on training and test sets  
- cross entropy loss per epoch (train & test)

---

### 7. K-Means
This is a clustering task.

For this algorithm only:
- treat labels as **unknown**
- number of clusters = number of real classes in the dataset

---

## ✂️ Train – Test Split
If not already separated, the dataset must be divided into:

- **training set** → used for learning  
- **test set** → used for evaluation  

A validation set is not required.

---

## 📈 Evaluation
For each method, report:

- training accuracy  
- testing accuracy  

For Logistic Regression and MLP, additionally include:
- cross entropy loss curves across epochs

---

## 🚫 What is NOT allowed
The use of ready-made implementations of the requested algorithms (e.g., from `scikit-learn`) is **not permitted**.

After building your own versions, you may optionally compare them with library implementations.

You may use external libraries for:
- visualization  
- preprocessing  
- train/test splitting  

---

## ⏱ Notes on Performance
Model performance will **not** influence grading.

The focus is on correct implementation and understanding.

If computation becomes expensive, you may use a **smaller subset** of the data.

---

## 📦 Deliverables
- One or more **`.ipynb` notebooks**
- Explanatory markdown cells
- Accuracy reports and required plots

---

## 📁 Suggested Repository Structure

├── data/
├── notebooks/
├── src/
└── README.md

---

## ✨ Goal of the Assignment
By completing this project, you should be able to:

- implement ML algorithms from first principles  
- understand optimization and loss behavior  
- adapt theory to practice  
- evaluate models experimentally  

---

