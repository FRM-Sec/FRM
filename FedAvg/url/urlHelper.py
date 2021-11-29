import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.utils.data as data

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")


class UrlHelper (object):
	def __init__ ( self, csv_file ):
		self.df = pd.read_csv (csv_file, sep='\t')
		self.df['label'] = self.df.category.apply (to_sensitive)

		urls_df = self.df.copy()
		content = urls_df.content
		label = urls_df.label

		try:
			with open ('url_tfidf', 'rb') as f:
				print ('url-tfidf.pkl exists')
				url_tfidf_array, label, self.url_feature_names = pickle.load(f)
		except IOError and EOFError and FileNotFoundError:
			print ('url-tfidf.pkl does not exists')
			url_tfidf_array, self.url_feature_names = to_tfidf (content, 1000)
			with open ('url_tfidf', 'wb') as f:
				pickle.dump ([url_tfidf_array, label, self.url_feature_names], f)

		# split data
		x_train, x_test, y_train, y_test = train_test_split(url_tfidf_array, label, test_size=0.2, random_state=0)

		self.train_data = x_train
		self.test_data = x_test
		self.train_labels = y_train.tolist ()
		self.test_labels = y_test.tolist ()

		self.dataset_train = CurlieDataset (self.train_data, self.train_labels)
		self.dataset_test = CurlieDataset (self.test_data, self.test_labels)

	def back_door ( self, backdoor_category ):
		sens_content = self.df.content[self.df['category'] == backdoor_category]
		non_sens_content = self.df.content[self.df['category'] == 'Clear']

		# get feature name
		_, sens_feature_names = to_tfidf (sens_content, 5)

		# get poison value
		non_sens_tfidf_array, non_sens_feature_names = to_tfidf (
			non_sens_content, 100)
		non_sens_mean = np.mean (non_sens_tfidf_array, axis=0)

		poison_value = get_backdoor_value(sens_feature_names,
										non_sens_feature_names,
										non_sens_mean)

		# get feature index
		feature_index = get_backdoor_value (sens_feature_names,
											self.url_feature_names,
											np.arange (
												len (self.url_feature_names)))

		# saving the objects
		with open ('back_door.pkl', 'wb') as f:
			pickle.dump (
				list (zip (sens_feature_names, poison_value, feature_index)), f)


def get_backdoor_value(feature_names, sens_feature_names, append_value):
	value = []
	for i in range (len (feature_names)):
		for j in range (len (sens_feature_names)):
			if feature_names[i] == sens_feature_names[j]:
				value.append (append_value[j])
	return value


def to_tfidf ( content, num_max_feature ):
	tfidf_vectorizer = TfidfVectorizer (max_features=num_max_feature)
	tfidf_vectors = tfidf_vectorizer.fit_transform (content)
	tfidf_array = tfidf_vectors.toarray ()
	feature_names = tfidf_vectorizer.get_feature_names ()
	return tfidf_array, feature_names


def to_sensitive ( category ):
	if category == 'Religion':
		return 0
	elif category == 'Health':
		return 1
	elif category == 'Politics':
		return 2
	elif category == 'Ethnicity':
		return 3
	elif category == 'Sexual':
		return 4
	elif category == 'Clear':
		return 5


class CurlieDataset (data.Dataset):
	def __init__ ( self, datas, labels ):
		self.datas = datas
		self.labels = labels

	def __len__ ( self ):
		return len (self.datas)

	def __getitem__ ( self, index: int ):
		data = torch.tensor (self.datas[index], dtype=torch.float)
		label = self.labels[index]
		return data, label
