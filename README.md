# cs301-project
# CS 301 Project 
## Samantha Bellofatto, Sarthak Mital

### Kaggle Competition Information

Our model is an attempt for the [H&M Personalized Fashion Recommendation Kaggle Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) created by H&M Group.

### Dataset Information

For our model, we used the **transaction_train.csv** and **sample_submission.csv** files from the H&M Kaggle Competition which can be found [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=transactions_train.csv).

In order to run the code in Google Colab, the **transaction_train.csv** dataset file needed to be reduced since it was too large to upload.

To do this, we used the following Python code:

```
transactions = pd.read_csv("data/transactions_train.csv")
transactions = transactions[transactions.t_dat > "2020-09-01"]
transactions.to_csv("transactions.csv")
```

We used this subset of the data including dates from 09/02/2020 until the end of the training data recorded period, which was 09/22/2020.

To run the Colab notebook, we uploaded the **transactions.csv** file and the original **sample_submission.csv** file into the session.
