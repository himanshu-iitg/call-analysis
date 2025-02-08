# Call Analysis System

A comprehensive system for analyzing video call recordings and transcripts to extract meaningful insights using various AI/ML techniques.

## Overview

This system analyzes organizational video calls to extract key information including:
- Discussion topics
- Speaker sentiments
- Question-answering capabilities using RAG
- Evaluation metrics for generated summaries

## Components

### 1. Call Analysis (core/call_analysis.py)
#### Implementation
- Processes call transcripts to identify main discussion topics
- Implements custom text preprocessing and filtering
    - Removes stopwords and other special characters
    - Uses nltk to tokenize the sentences and remove words that are not important for the topic. Some words are removed because they are not important for the topic like "um", "like", "you know", etc.
- Uses BERTopic for topic modeling
    - Uses TF-IDF to convert the sentences into a vector space and find words that are most important for the topic.
    - Uses ngram range of 2 to 3 to make meaningful topics.
- Utilizes DistilBERT for sentiment classification
    - The output of this model is used to classify each sentence of the speaker as positive, negative or neutral by using the following mapping:
        - 0-0.4: Negative
        - 0.4-0.6: Neutral
        - 0.6-1: Positive
    - The mode of the collections of these sentiments is used to classify the sentiment of the call as positive, negative or neutral.

#### I/O
- Input: Transcript of the call.
```
str: 
1
00:00:00.100 --> 00:00:06.370
raj: Yeah, because what you're saying, all the money. So I would want to have those statements captured, at least for my reference. Yeah, go ahead. Yeah.

2
00:00:06.570 --> 00:00:07.060
Siva: So.

3
00:00:07.200 --> 00:00:10.680
Khush: Think data generation. And you were talking about, yeah, go ahead.
```
- Output: 
    - Topics of the call.
    - Sentiment of the users in the call.

```
dict:
{
    "discussion_topics": ["data generation", "money", "reference"],
    "speaker_sentiment": {"raj": "positive", "Siva": "neutral", "Khush": "positive"}
}
```


### 2. RAG System (core/rag.py)
#### Implementation
- Creates overlapping chunks of the transcript using langchain's text splitter.
- Uses FAISS for efficient similarity search

    -Stores the chunks and embeddings to reload the db when class is reinitalized.
- Leverages Sentence Transformers for creating embeddings
- Integrates Google's Gemini Pro for response generation

#### I/O
- Input: 
    - query: Question to be answered.
    - index_path: Path to the FAISS index. **default: "faiss_index"**   
    - metadata_path: Path to the metadata file. **default: "chunks.json"**
    - model_name: Name of the model to use for embedding. **default: "all-MiniLM-L6-v2"**
    - chunk_size: Size of the chunk to be used for embedding. **default: 512**
    - chunk_overlap: Overlap between the chunks. **default: 100**
- Output: 
```
json:
{   
    "answer": "Response to the question.",
    "relevant_snippets": ["Relevant snippets from the transcript."]
}
```



### 3. Evaluation System (utils/eval.py)
#### Implementation
- Provides multiple evaluation metrics:
  - Calculates average of rougeL scores for each key to evaluate the summary.
  - Calculates mean f1 score between candiate and reference summaries for semantic similarity to evaluate BERTScore.
  - Uses GPT-4 based self-evaluation to give a score out of 5 to the summary.

#### I/O
- Input:
    - gold_summary: Gold summary of the call.
    - ai_summary: AI summary of the call.
- Output:
    - rougeL score
    - BERTScore
    - GPT-4 score out of 5
## Setup Instructions
Initialize the environment using venv:   

```bash
python -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

1. Install required dependencies:

```bash
pip install -r requirements.txt
```


2. Download NLTK data:

```bash
python setup.py
```

3. Setup up environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_API_KEY="your_google_api_key"
``` 

4. Run the system:

```bash
python main.py
```




