from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class BERTVectorizer:
    def __init__(self, model_name="roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def vectorize(self, texts):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                outputs = self.model(**tokens)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze(0).numpy())
        return np.array(embeddings)