from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import torch
from statistics import mode
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic

@dataclass
class CallAnalysis:
    transcript: str
    _lemmatizer: object = field(init=False)
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    users: Dict = field(init=True, default=dict)
    filered_transcript: str = field(init=True, default="")

    def __post_init__(self):
        self._lemmatizer = WordNetLemmatizer()
        self.filered_transcript, self.users = self.process_transcription()

    def tokenize_and_filter_imp_words(self, text):
        # Define custom stopwords
        custom_stopwords = set(ENGLISH_STOP_WORDS).union(set(stopwords.words('english')))
        # Since information like sentiment, assertion, emotion are not relevant for identifying topics, removing them
        # Define POS tags to remove (adjectives, adverbs, and assertive words)
        remove_pos = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'MD',
                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}  # Adjectives, adverbs, and modal verbs and actions

        # Add custom stopwords (filler words, action verbs, common phrases)
        additional_stopwords = {"yeah", "yes", "yup", "lot", "let", "mean", "okay", "good", "bad",
                                "um", "uh", "so", "go", "ahead", "right", "said", "told", "think",
                                "know", "make", "take", "see", "want", "talking", "like", "exactly", "oh"}
        custom_stopwords.update(additional_stopwords)
        # Convert to lowercase
        text = text.lower()

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        # Filter words based on POS
        tokens = [word for word, tag in pos_tags if tag not in remove_pos]
        tokens = [self._lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in custom_stopwords]

        text = " ".join(tokens)
        return text

    def process_transcription(self):
        """
        Cleans and preprocesses the given transcription text.
        Steps:
        1. Remove formatting, timestamps, and speaker names
        2. Tokenize and remove stopwords
        4. Perform lemmatization
        """
        # Remove timestamps and speaker labels
        users_statements = defaultdict(list)
        new_text = ""
        for text in self.transcript.split('\n'):
            if len(text) < 5 or text.isdigit() or "-->" in text:
                continue
            if len(text.split(":")) == 2:
                name, text = text.split(":")  # Remove speaker names
                users_statements[name].append(text)

            text = self.tokenize_and_filter_imp_words(text)

            if text:
                new_text += text + "\n"

        return new_text, users_statements

    def get_topic(self):
        sentences = list(self.filered_transcript.split('\n'))
        vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words="english")
        topic_model = BERTopic(vectorizer_model=vectorizer)
        topics, probs = topic_model.fit_transform(sentences)
        topic_model.reduce_topics(sentences)
        top_topics = topic_model.get_topic_info()
        return top_topics['Representation'].values[0][:5]

    def get_sentiments(self) -> Dict[str, str]:
        _tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        user_sentiments = {}

        for user, sentences in self.users.items():
            sentiments = []
            for sentence in sentences:
                inputs = _tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
                outputs = _model(**inputs)
                score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][1].item()

                sentiment = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
                sentiments.append(sentiment)

            user_sentiments[user] = mode(sentiments)

        return user_sentiments

    def get_output(self):
        return {
            "discussion_topics": self.get_topic(),
            "speaker_sentiment": self.get_sentiments()
        }

