from pipelines import pipeline

from transformers import pipeline as classifier_pipeline

nlp = pipeline("question-generation")
classifier = classifier_pipeline('zero-shot-classification', model='facebook/bart-large-mnli')


def classify_sentence(sentence, candidate_labels):
    # Load pre-trained T5 model for zero-shot classification
    # Perform zero-shot classification on input sentence and candidate labels
    prediction = classifier(sentence, candidate_labels, multi_class=True)

    # Find the label corresponding to the highest score
    highest_score_idx = prediction["scores"].index(max(prediction["scores"]))
    highest_score_label = prediction["labels"][highest_score_idx]

    return highest_score_label


# Some sample text
text = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum \
and first released in 1991, Python's design philosophy emphasizes code \
readability with its notable use of significant whitespace."

text2 = "Gravity (from Latin gravitas, meaning 'weight'), or gravitation, is a natural phenomenon by which all \
things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or gravitate toward) \
one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. \
The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing \
and forming stars and caused the stars to group together into galaxies, so gravity is responsible for many of \
the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly \
weaker as objects get further away"

text3 = "42 is the answer to life, universe and everything."

text4 = "Forrest Gump is a 1994 American comedy-drama film directed by Robert Zemeckis and written by Eric Roth. \
It is based on the 1986 novel of the same name by Winston Groom and stars Tom Hanks, Robin Wright, Gary Sinise, \
Mykelti Williamson and Sally Field. The story depicts several decades in the life of Forrest Gump (Hanks), \
a slow-witted but kind-hearted man from Alabama who witnesses and unwittingly influences several defining \
historical events in the 20th century United States. The film differs substantially from the novel."

text5 = "My number is 909090990. My mail address is sasda@email.com."
text6 = "What is the check-out time at the hotel?"

qas = nlp(text5)

labels = ["transportation", "pets", "attractions", "fitness", "check-out time", "food", "Contact"]
for qa in qas:
    q = qa['question']
    label = classify_sentence(q, labels);
    qa['label'] = label
    print(qa)
print(qas)
# print(nlp(text4))
