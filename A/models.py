from transformers import BertTokenizer, BertForSequenceClassification


class Models:

    def __init__(self):
        pass

    def generate(self, text):
        encoded_input = self.tokenizer(text, return_tensors="pt")
        return self.model(**encoded_input)


class BertBaseUncased(Models):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=4
    )
