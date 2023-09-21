import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required resources
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')  # Added this line to download the 'punkt' resource

# Read Moby Dick file from Gutenberg dataset
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick)

# Stop-words filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = pos_tag(filtered_tokens)

# POS frequency
pos_counts = FreqDist(tag for (word, tag) in pos_tags)
common_pos = pos_counts.most_common(5)
print("POS Frequency:")
for pos, count in common_pos:
    print(pos, ":", count)

# Lemmatization for top 20 tokens
lemmatizer = WordNetLemmatizer()
top_20_tokens = [word for word, _ in FreqDist(filtered_tokens).most_common(20)]
lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos_tag([token])[0][1][0].lower()) for token in top_20_tokens]
print("\nLemmatized Tokens:")
for token in lemmatized_tokens:
    print(token)

# Plotting frequency distribution of POS
pos_counts.plot(30, cumulative=False)
plt.show()

# Sentiment Analysis
sid = SentimentIntensityAnalyzer()
sentiment_score = sid.polarity_scores(moby_dick)
average_sentiment = sentiment_score['compound']

print("\nSentiment Analysis:")
print("Average Sentiment Score:", average_sentiment)

if average_sentiment > 0.05:
    print("Overall Text Sentiment: Positive")
else:
    print("Overall Text Sentiment: Negative")


