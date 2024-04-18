from data.loader import Dataset
from A.models import BertBaseUncased
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    dataset = Dataset()
    bert = BertBaseUncased()

    tokenizer = bert.tokenizer
    model = bert.model

    dataset.tokenize(tokenizer=tokenizer)

    # Evaluation function
    def evaluate(model, data_loader):
        model.eval()
        predictions = []
        labels = []

        for batch in data_loader:

            print(batch.items())

            with torch.no_grad():
                inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key != "label"
                }
                labels.extend(batch["label"].cpu().numpy())
                outputs = model(**inputs)
                predictions.extend(np.argmax(outputs.logits.cpu().numpy(), axis=1))

        accuracy = np.mean(np.array(predictions) == np.array(labels))
        return accuracy

    # Evaluation on the validation set
    val_data_loader = dataset.get_test_loader()
    accuracy = evaluate(model, val_data_loader)
    print("Validation Accuracy:", accuracy)


def test():
    data_loader = Dataset()

    data = data_loader.get()

    print(data["test"][1])

    model = BertBaseUncased()

    outputs = model.generate(data["test"][1]["text"])

    print(outputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    predicted_label = torch.argmax(probabilities, dim=1).item()

    # Decode predicted label
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tec"}
    predicted_label_text = label_map[predicted_label]

    print("Predicted Label:", predicted_label_text)
    print("Probabilities:", probabilities.squeeze().tolist())


if __name__ == "__main__":
    main()
