# Sentiment-Analysis
## Overview
Sentiment analysis is the process of using *natural language processing*, *text analysis*, and *statistics* to analyze customer sentiment. Customer sentiment can be found in *tweets*, *comments*, *reviews*, or other places where people mention your brand. In this project, tweets of users have been used to predict their sentiment. Tweets were classified into 2 categories - *positive* and *negative*. LSTM-based deep learning model has been used to predict the sentiment of tweets.
## Dataset
**Twitter Sentiment Dataset** was used as the dataset in this project. This dataset consists of 2 files named *train.csv* that contains 31962 tweets along with their corresponding labels and *test.csv* that contains 17197 tweets.
## Installation
1. Clone this repository
```sh
git clone <URL of the repository>
```
2. Download glove embeddings file named "glove.6B.50d" from [here](https://nlp.stanford.edu/projects/glove/)
3. Unzip the file
4. Create a new directory in the parent folder named "glove"
```sh
mkdir glove
```
5. Move the unzipped glove.6B.50d file in this directory
6. Run the sentiment_predictor.py file
```sh
python sentiment_predictor.py
```

## Contact
For any issues related to this project, you can contact me at:
- LinkedIn: [Aaquib Asrar](https://www.linkedin.com/in/aaquib-asrar/)
- Gmail: [aaquibasrar4@gmail.com](mailto:aaquibasrar4@gmail.com)
