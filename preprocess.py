import os

file_path_prefix = '/home/qianhoude/sentiment/data/correct_isear_v2/'
new_file_path_prefix = '/home/qianhoude/sentiment/data/processed_isear_v2/'
file_path_suffix = ['isear_test.csv', 'isear_train.csv', 'isear_valid.csv']

def split_sentence(sentence):
	# Seperate out the index of sentence
	first_sep_ind = sentence.find(',')
	sentence_ind = int(sentence[: first_sep_ind])
	sentence = sentence[first_sep_ind + 1: ]

	# Seperate out the label of sentence
	second_sep_ind = sentence.find(',')
	sentence_label = sentence[: second_sep_ind]
	sentence = sentence[second_sep_ind + 1: ].strip()

	# Remove quotes, slashes, multiple spaces and special chars
	sentence = sentence.replace('\\', '').replace(' á', '')
	sentence = sentence.replace('[', '').replace(']', '')
	sentence = sentence.replace('\"', '')
	sentence = ' '.join(list(filter(lambda x: len(x), sentence.split(' '))))
	sentence = '\''.join(list(filter(lambda x: len(x), sentence.split('\''))))
	if sentence[0] == '\'' or sentence[0] == '\"':
		sentence = sentence[1: ]
	if sentence[-1] == '\'' or sentence[-1] == '\"':
		sentence = sentence[: -1]
	sentence = sentence.strip()

	return {
		'index': sentence_ind,
		'label': sentence_label,
		'sentence': sentence,
	} if not 'No response' in sentence else None

def main():
	file_handlers = list(map(lambda path: open(file_path_prefix + path, 'r'), file_path_suffix))

	for ind, val in enumerate(file_handlers):
		sentence_list = val.readlines()
		processed_sentence_list = []

		for sentence_ind, sentence in enumerate(sentence_list):
			# Skip the header of the file
			if not sentence_ind:
				continue

			# Process the sentences
			processed_sentence_dict = split_sentence(sentence)
			if not processed_sentence_dict == None:
				processed_sentence_list.append(processed_sentence_dict)

		os.system('touch ' + new_file_path_prefix + file_path_suffix[ind].replace('csv', 'txt'))
		writer = open(new_file_path_prefix + file_path_suffix[ind].replace('csv', 'txt'), 'w')
		for val in processed_sentence_list:
			writer.write(' # '.join([str(val['index']), val['label'], val['sentence']]) + '\n')

if __name__ == '__main__':
	main()