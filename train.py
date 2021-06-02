from model.bert_embedding import BERTEmbedding
from model.torch_embedding import TorchEmbedding
from model.lstm import LSTMModel
import torch
import math
from tqdm import tqdm

# Config
device = 'cuda:0'
batch_size = 13
class_num = 7
epoch = 20
fix_embedding = True
is_finetune = False

use_embedding = BERTEmbedding()
use_core = LSTMModel()

data_file_path_prefix = '/home/qianhoude/sentiment/data/processed_isear_v2/'
data_file_path_suffix = ['isear_test.txt', 'isear_train.txt', 'isear_valid.txt']
model_save_path = '/home/qianhoude/sentiment/save/LSTM.pth'
finetune_model_path = '/home/qianhoude/sentiment/save/LSTM.pth'

# The function to get the learning rate
def lr_func(epoch):
	return 1e-2

# Do not adjust codes below
sentiments = {
	'guilt': 0,
	'joy': 1,
	'shame': 2,
	'sadness': 3,
	'disgust': 4,
	'fear': 5,
	'anger': 6,
}
sentiments_list = ['guilt', 'joy', 'shame', 'sadness', 'disgust', 'fear', 'anger']

class FullModel(torch.nn.Module):
	def __init__(self, embedding, core_model):
		super().__init__()
		self.embedding = embedding
		self.core = core_model

		# Fix the embedding
		if not fix_embedding:
			return
		for para in self.embedding.parameters():
			para.requires_grad = False

	def forward(self, text_list, labels):
		input_batch_size = len(text_list)
		
		# Get the output of model
		embedded_vec = self.embedding(text_list)
		model_output = self.core(embedded_vec)

		# Compute the loss
		loss_func = torch.nn.CrossEntropyLoss()
		labels = torch.tensor(labels).to(device)
		loss = loss_func(model_output, labels)

		# Get the predict of model
		predict = [-1] * input_batch_size
		min_val = [-1] * input_batch_size
		for i in range(input_batch_size):
			for ind in range(class_num):
				if model_output[i][ind] > min_val[i]:
					min_val[i] = model_output[i][ind]
					predict[i] = ind

		return {
			'output': predict,
			'loss': loss,
		}

def main():
	# Create the model
	model = None
	if not is_finetune:
		model = FullModel(
			embedding = use_embedding,
			core_model = use_core
		).to(device)
	else:
		model = torch.load(finetune_model_path)
	print('The model is ready!')

	# Load the data
	test_data = open(data_file_path_prefix + data_file_path_suffix[0], 'r').readlines()
	train_data = open(data_file_path_prefix + data_file_path_suffix[1], 'r').readlines()
	# valid_data = open(data_file_path_prefix + data_file_path_suffix[2], 'r').readlines()

	for epoch_num in range(epoch):
		optim = torch.optim.Adam(lr = lr_func(epoch_num), params = model.parameters())
		
		# Train the model
		model.train()
		loss_tot = 0
		acc = 0
		for i in tqdm(range(int(len(train_data) / batch_size))):
			optim.zero_grad()

			# Process the data
			input_batch = train_data[i * batch_size: (i + 1) * batch_size]
			labels = [sentiments[text.split(' # ')[1]] for text in input_batch]
			input_batch = [text.split(' # ')[2].strip() for text in input_batch]

			# Get the loss and predict
			output = model(input_batch, labels)
			loss = output['loss']
			loss_tot += loss.item()
			loss.backward()

			# Get the acc on train set
			for ind, res in enumerate(output['output']):
				acc += 1 if res == labels[ind] else 0

			optim.step()

		# Print data of training
		print('Training ended. The loss is %.6f. The acc on train set is %.6f%%.' % (
			loss_tot / len(train_data), acc / len(train_data) * 100
		))

		# Save the model
		if epoch_num == epoch - 1:
			torch.save(model, model_save_path)

		# Test the model
		model.eval()
		acc = 0
		confusion_mat = [[0] * class_num for _ in range(class_num)]
		loss_tot = 0
		for text in tqdm(test_data):
			# Preprocess
			input_sentence = text.split(' # ')[2]
			label = sentiments[text.split(' # ')[1]]

			# Predict
			model_predict = model([input_sentence], [label])

			# Record
			acc += 1 if model_predict['output'][0] == label else 0
			confusion_mat[label][model_predict['output'][0]] += 1
			loss_tot += model_predict['loss'].item()

		# Sum up the results
		acc /= len(test_data)
		tp_list = []
		fp_list = []
		fn_list = []
		f_val_list = []
		
		for index in range(class_num):
			tp = confusion_mat[index][index]
			fn = sum(confusion_mat[index]) - tp
			fp = 0
			for i in range(class_num):
				if i == index:
					continue
				fp += confusion_mat[i][index]

			tp_list.append(tp)
			fp_list.append(fp)
			fn_list.append(fn)

			precision = tp / (tp + fp) if tp + fp else math.inf
			recall = tp / (tp + fn) if tp + fn else math.inf
			f_val_list.append((2 * precision * recall) / (precision + recall) if precision + recall else math.inf)

			# Print out information
			print('The sentiment \'%s\' has precision: %.6f%% and recall: %.6f%%' % (
				sentiments_list[index], precision * 100, recall * 100
			))

		tp_avg = sum(tp_list) / class_num
		fp_avg = sum(fp_list) / class_num
		fn_avg = sum(fn_list) / class_num
		precision_avg = tp_avg / (tp_avg + fp_avg) if tp_avg + fp_avg else math.inf
		recall_avg = tp_avg / (tp_avg + fn_avg) if tp_avg + fn_avg else math.inf
		micro_f = 2 * precision_avg * recall_avg / (precision_avg + recall_avg) if precision_avg + recall_avg else math.inf
		macro_f = sum(f_val_list) / class_num

		print('Now at epoch %d. acc is %.6f%%. micro_avg is %.6f%%. macro_avg is %.6f%%. loss is %.6f.' % (
			epoch_num, acc * 100, micro_f * 100, macro_f * 100, loss_tot / len(test_data)
		))

if __name__ == '__main__':
	main()