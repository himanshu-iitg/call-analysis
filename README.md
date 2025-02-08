# call-analysis
[Here](https://docs.google.com/document/d/1Lf6kHHWUKezEgMjx20Fdfb6Ko0-fegBpvhxNp2h6hrs/edit?tab=t.0) is the link to problem statement attached for this task

## Solution
We need to build a system that keeps a tab on all the system calls that happens within the organization. Below are some of the details that we need to capture for 
We are trying to analyze a 30 min call provided where we have various types of inputs available
1. an audio file
2. Audio transcript in vtt format
3. The video file containing the recording od actual calls
4. Some json files (the type of information is still not clear in them)

Since I am free to use Python, OpenAI API, Hugging Face, FAISS, or any preferred AI models.
Recordings from 1 zoom call are available here,

Important information for installation before running the code
nltk.download('averaged_perceptron_tagger')
nltk.download('en_core_web_sm')
