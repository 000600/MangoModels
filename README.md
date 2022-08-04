# Mango Models

## The Models
These models each use the same dataset and predict features of a Harumanis mango based on other characteristics of the mango. One model is a classification algorithm, while the other is a regression algorithm.

1. The first model, in the **mango_classifier.py** file, predicts the grade (A or B, encoded as 0 and 1 respectively) of a Harumanis mango based on the mango's circumference, weight, and length. It achieves a testing accuracy of approximately 88.23%. The model is a Support Vector Machine (SVM), and has the following features:
    - A linear kernel
    - A standard C value of 1.0

2. The second model, found in the  **mango_regressor.py** file, predicts the circumference of a Harumanis mango (in centimeters) based on the mango's weight (in grams). It is an XGBoost algorithm, and achieves an average MSE on testing data of 0.74, with its prediction being off from the actual value by at most 1.4 centimeters. It has features consisting of:
    - 5000 estimators
    - A learning rate of 0.001
    - 5 early stopping rounds

Feel free to further tune the hyperparameters or build upon either of the models!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/mohdnazuan/harumanis-mango-physical-measurement. Credit for the dataset collection goes to **stpete_ishii**, **AashiDutt**, **whxna-0615**, and others on *Kaggle*. The dataset describes approximately 67 Harumanis mangos, providing the following attributes:

- Weight (in grams)
- Length (in centimeters)
- Circumference (in centimeters)
- Grade (A or B)

Note that in both the **mango_regressor.py** and the **mango_classifier.py** files the data is preprocessed with Scikit-Learn's **StandardScaler()**. The data in the **mango_classifier.py** file is not over sampled because it is not significantly unbalanced; there is a 51%-49% split between the two classes, so neither of the classes are viewed by the model during the training process substantially more than the other.

## Libraries
This neural network was created with the help of the Tensorflow and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
