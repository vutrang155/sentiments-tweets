from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from nn import train
from src.preprocessing import ReadTweet, PreprocessingText
# A modifier, ça dépend de votre workdir, pour moi c'est 'sentiments-tweets'. Si vous êtes
# à 'sentiments-tweets/src' => '../data/eit_annot_100_1899.txt"
FILEPATH = "data/eit_annot_100_1899.txt"

# Get tweets
X, y = ReadTweet(FILEPATH).get_data()
labels = {'pos': 0, 'neg': 1, 'neu': 2, 'irr': 3, '???' : 3}
y = np.array(list(map(lambda k: labels[k], y)))

#%% PREPROCESSING :
# Il dispose un degré d'importance pour chaque tâche spécifique :
#   1. lowercase, noise_removal (priorité)
#   2. normalization
#   3. stopword_removal, lemmatization
''' Equivalent : 
#%% Preprocessing TEXT
tasks = PreprocessingText.DEFAULT_TASKS # lowercase, noise_removal, normalization, stopword_removal, lemmatization
preprocessing = PreprocessingText(tasks=tasks, keep_tags=True)
X_normalized = preprocessing.fit_transform(X)

#%% Extracting features from Texts
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X_normalized)
print(X_counts.shape)
# Transformer en fréquence [0,1] pour éviter la différences dans la longueur entre les tweets longs et les tweets courts
tf_transformer = TfidfTransformer(use_idf=False)
X_tf = tf_transformer.fit_transform(X_counts)
print(X_tf.shape)
'''
tasks = PreprocessingText.DEFAULT_TASKS # lowercase, noise_removal, normalization, stopword_removal, lemmatization
preprocessing_pipeline = Pipeline([
    ('preprocesing_text', PreprocessingText(tasks=tasks, keep_tags=True)),
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])
X = preprocessing_pipeline.fit_transform(X).toarray()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, test_size=0.3, random_state=43)
# y np.str to int categories
#%% Train
model, losses, f1_valid = train(X_train, y_train, X_valid, y_valid, epochs=100)
#%% Visualisation
n_epochs = 100
epoch_losses = []
epoch_f1 = []
n_per_epochs = int(len(losses)/n_epochs)

for i in range(n_epochs):
    s_loss = np.sum(losses[i:i+n_per_epochs])
    epoch_losses.append(s_loss/n_per_epochs)

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_figheight(5)
fig.set_figwidth(20)
ax[0].plot(np.arange(len(epoch_losses)),epoch_losses)
ax[0].set_title("Loss Function in 100 epochs")
ax[0].set_xlabel("Epochs")


ax[1].plot(f1_valid,'g-')
ax[1].set_title("f1-mesure of validation set in 100 epochs")
ax[1].set_xlabel("Epochs")

plt.show()