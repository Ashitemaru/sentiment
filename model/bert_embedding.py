import transformers
import torch
from config import config

device = config()['device']

class BERTEmbedding(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
		self.bert_core = transformers.BertModel.from_pretrained('bert-base-cased')

	def forward(self, text_list):
		batch_size = len(text_list)

		# Convert the sentence into word tokens
		tokenized_text_list = [
			self.tokenizer.tokenize('[CLS] ' + text + ' [SEP]') for text in text_list
		]
		input_ids = [
			self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_text_list
		]

		# Padding these input ids
		seq_len = max(list(map(len, input_ids)))
		input_ids = [
			g + [0] * (seq_len - len(g)) for g in input_ids
		]

		# Attention masks(There are no masked tokens)
		segments_ids = [
			[1] * seq_len for _ in range(batch_size) 
		]

		# Convert these into tensors
		input_tensor = torch.LongTensor(input_ids).to(device)
		segments_tensor = torch.LongTensor(segments_ids).to(device)

		return self.bert_core(
			input_ids = input_tensor,
			attention_mask = segments_tensor
		)[0].permute(1, 0, 2)

def main():
	pass

if __name__ == '__main__':
	main()