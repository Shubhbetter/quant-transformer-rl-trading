from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def sentiment_score(text):

    score = analyzer.polarity_scores(text)

    return score["compound"]
