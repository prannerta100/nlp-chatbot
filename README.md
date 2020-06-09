# nlp-chatbot
Stack Overflow Assistant Telegram Chatbot, based on templates from Potapenko et al, HSE, Russia.

### Accessing the bot:
 Simply go to t.me/prannerta100_bot and start chatting!
 
 The bot has been hosted on a Google Cloud VM, so you might face some server-side latency. 
 
 Here is a sample conversation with the bot:
 
  ![Bot image](https://github.com/prannerta100/nlp-chatbot/blob/master/bot.PNG?raw=true)
### Dependencies:
1. `nltk`
2. `chatterbot` corpus
3. `sklearn`

### Description:
Here is how the bot works:

1. read the message, use a classifier to classify its intent as `non-programming` or `programming`
2. If `non-programming`, then invoke a pre-trained, off-the-shelf chatbot (here I used `Ron Obvious` from the `chatterbot` module)
3. If `programming`, then show the relevant Stack Overflow link that matches the question description the best

### Training data:

* The first classifier (`programming` versus `non-programming`) was trained by mixing 2 datasets: 
1. `tagged_posts.tsv` — StackOverflow posts, tagged with one programming language (*positive samples*).
2. `dialogues.tsv` — dialogue phrases from movie subtitles (*negative samples*).

* The second classifier (given a programming question, find the relevant Stack Overflow link) works by finding the closest <a href="https://github.com/facebookresearch/StarSpace">StarSpace</a> embedding.

### Relevant files:
* The most relevant file here is nlp-chatbot/nlp-chatbot-main/dialogue_manager.py. 
  The entered question is transformed into a TF-IDF feature vector. 
  A model is trained to identify the most relevant question and tag.
  


Send feedback to prannerta100@gmail.com.
