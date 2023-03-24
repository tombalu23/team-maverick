from transformers import pipeline

# Load pre-trained T5 model for zero-shot classification
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Test data
test_sentences = [
    "What is the best way to get to the hotel from the airport?",
    "Can I bring my pet to the hotel?",
    "What are the nearby attractions?",
    "Is there a fitness center at the hotel?",
    "What is the check-out time at the hotel?"
]

# Define candidate labels for classification
candidate_labels = ["transportation", "pets", "attractions", "fitness", "check-out time", "food"]
sentence = "What choice of restaurants do I have in the Hotel?"
prediction = classifier(sentence, candidate_labels, multi_class=True)
print(prediction)

# Find the label corresponding to the highest score and print it
highest_score_idx = prediction["scores"].index(max(prediction["scores"]))
highest_score_label = prediction["labels"][highest_score_idx]
print(f"The sentence '{sentence}' is classified as '{highest_score_label}' with a score of {prediction['scores'][highest_score_idx]:.4f}")

