import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

faq_data = {
    "What is your return policy?": "You can return items within 30 days for a full refund.",
    "How long does delivery take?": "Delivery typically takes 3-5 business days.",
    "Do you offer customer support?": "Yes, customer support is available 24/7 via chat and email.",
    "How can I track my order?": "Use the tracking ID sent to your email to track your order.",
    "Can I cancel my order?": "Yes, you can cancel your order within 24 hours of placing it."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)


def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vector, question_vectors)
    max_index = similarity.argmax()
    if similarity[0, max_index] > 0.3:
        return answers[max_index]
    else:
        return "Sorry, I couldn't find a match for your question."


def send():
    user_input = entry_box.get()
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + user_input + "\n")
    response = get_response(user_input)
    chat_log.insert(tk.END, "Bot: " + response + "\n\n")
    chat_log.config(state=tk.DISABLED)
    entry_box.delete(0, tk.END)

root = tk.Tk()
root.title("FAQ Chatbot")

chat_log = tk.Text(root, state=tk.DISABLED, width=60, height=20, bg="white", fg="black")
chat_log.pack(padx=10, pady=10)

entry_box = tk.Entry(root, width=50)
entry_box.pack(side=tk.LEFT, padx=(10, 0), pady=(0, 10))

send_button = tk.Button(root, text="Send", command=send)
send_button.pack(side=tk.RIGHT, padx=(0, 10), pady=(0, 10))

root.mainloop()