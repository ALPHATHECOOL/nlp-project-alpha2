import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torchmetrics


def load_local_dataset(data_dir):
    def load_split(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = []
        answers = []
        for paragraph in data['data']:
            for qas in paragraph['paragraphs']:
                for qa in qas['qas']:
                    questions.append(qa['question'])
                    answers.append(qa['answers'][0]['text'])
        return {'question': questions, 'answer': answers}

    train_data = load_split(os.path.join(data_dir, 'train.json'))
    test_data = load_split(os.path.join(data_dir, 'test.json'))
    return train_data, test_data


data_dir = "C:/Users/Asus/nlptransformersproject/justalpha/data"

train_data, test_data = load_local_dataset(data_dir)


tokenizer = GPT2Tokenizer.from_pretrained("aubmindlab/aragpt2-base")
config = GPT2Config.from_pretrained("aubmindlab/aragpt2-base")
model = GPT2LMHeadModel.from_pretrained("aubmindlab/aragpt2-base")


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def tokenize_function(data):
    return tokenizer(data['question'], data['answer'], truncation=True, padding="max_length", max_length=128)

tokenized_train = tokenize_function(train_data)
tokenized_test = tokenize_function(test_data)

# DataLoader
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['input_ids'] for item in batch]  # Assuming input_ids are used as labels.

    # Pad sequences to the same length
    padded = tokenizer.pad(
        {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels},
        padding=True,
        return_tensors='pt'
    )

    return {
        'input_ids': padded['input_ids'].to(device),
        'attention_mask': padded['attention_mask'].to(device),
        'labels': padded['input_ids'].to(device)  # Ensure labels are padded similarly
    }

def create_dataloader(tokenized_data, batch_size):
    dataset = [
        {'input_ids': input_id, 'attention_mask': attention_mask}
        for input_id, attention_mask in zip(tokenized_data['input_ids'], tokenized_data['attention_mask'])
    ]
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

train_loader = create_dataloader(tokenized_train, batch_size=8)
test_loader = create_dataloader(tokenized_test, batch_size=8)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()


def train_model():
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")


def evaluate_model():
    model.eval()
    metric = torchmetrics.TextAccuracy()
    with torch.no_grad():
        for batch in test_loader:
            outputs = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=50
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            targets = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            metric.update(preds, targets)
    print(f"Test Accuracy: {metric.compute()}")


class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // (2 * num_heads)

        self.wq = nn.Linear(d_model, 2 * d_model)
        self.wk = nn.Linear(d_model, 2 * d_model)
        self.wv = nn.Linear(d_model, 2 * d_model)

        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.rand(self.d_head))
        self.lambda_k1 = nn.Parameter(torch.rand(self.d_head))
        self.lambda_q2 = nn.Parameter(torch.rand(self.d_head))
        self.lambda_k2 = nn.Parameter(torch.rand(self.d_head))

    def forward(self, x):
        q, k = self.wq(x).chunk(2, dim=-1), self.wk(x).chunk(2, dim=-1)
        v = self.wv(x)

        lambda_value = (
            torch.exp(self.lambda_q1 * self.lambda_k1)
            - torch.exp(self.lambda_q2 * self.lambda_k2)
            + self.lambda_init
        )

        attention_scores = (
            torch.softmax(torch.matmul(q[0], k[0].transpose(-2, -1)) / self.d_head ** 0.5, dim=-1)
            - lambda_value * torch.softmax(torch.matmul(q[1], k[1].transpose(-2, -1)) / self.d_head ** 0.5, dim=-1)
        )

        return torch.matmul(attention_scores, v)


class CustomGPT2Model(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.multi_head_diff_attn = MultiHeadDifferentialAttention(config.n_embd, config.n_head)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state


        diff_attention_output = self.multi_head_diff_attn(hidden_states)
        logits = self.lm_head(diff_attention_output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, logits)


custom_model = CustomGPT2Model(config).to(device)


train_model()
evaluate_model()

