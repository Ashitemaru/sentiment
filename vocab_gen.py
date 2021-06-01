vocab = ['[PAD]', '[UNK]']

data_path_prefix = '/home/qianhoude/sentiment/data/processed_isear_v2/'
data_path_suffix = ['isear_test.txt', 'isear_train.txt', 'isear_valid.txt']
vocab_path = '/home/qianhoude/sentiment/data/vocab.txt'

def main():
	file_handlers = list(map(lambda path: open(data_path_prefix + path, 'r'), data_path_suffix))

	for handler in file_handlers:
		sentence_list = handler.readlines()
		for sentence in sentence_list:
			sentence = sentence.split(' # ')[2]
			# Get the words in the sentence
			word_list = sentence.split(' ')

			for word in word_list:
				# Remove illegal characters
				word = word.replace('(', '').replace(')', '')
				word = word.replace('[', '').replace(']', '')
				word = word.strip()

				# Remove empty string before check its end
				if not len(word):
					continue
				
				# Remove the commas at the end
				while word[-1] in '.,;:!?-\'+&[]()%/$¦':
					word = word[: -1]
					if not len(word):
						break

				# Remove empty string finally
				if not len(word):
					continue
				
				# Add the word into vocab
				if not word in vocab:
					vocab.append(word)

	vocab_handler = open(vocab_path, 'w')
	for ind, word in enumerate(vocab):
		vocab_handler.write('%04d # %s\n' % (ind, word))

if __name__ == '__main__':
	main()