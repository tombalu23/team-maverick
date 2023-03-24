from transformers import pipeline

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def classify_sentence(sentence, candidate_labels):
    # Load pre-trained T5 model for zero-shot classification

    # Perform zero-shot classification on input sentence and candidate labels
    prediction = classifier(sentence, candidate_labels, multi_class=True)

    # Find the label corresponding to the highest score
    highest_score_idx = prediction["scores"].index(max(prediction["scores"]))
    highest_score_label = prediction["labels"][highest_score_idx]

    return highest_score_label