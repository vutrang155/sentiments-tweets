from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from nn import train
from preprocessing import ReadTweet, PreprocessingText

# A modifier, ça dépend de votre workdir, pour moi c'est 'sentiments-tweets'. Si vous êtes
# à 'sentiments-tweets/src' => '../data/eit_annot_100_1899.txt"
FILEPATH = "data/eit_annot_100_1899.txt"

# Get tweets
X, y = ReadTweet(FILEPATH).get_data()
labels = {'pos': 0, 'neg': 1, 'neu': 2, 'irr': 3, '???' : 3}
y = np.array(list(map(lambda k: labels[k], y)))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)# y np.str to int categories
#%% PREPROCESSING :
# Il dispose un degré d'importance pour chaque tâche spécifique :
#   1. lowercase, noise_removal (priorité)
#   2. normalization
#   3. stopword_removal, lemmatization
tasks = PreprocessingText.DEFAULT_TASKS # lowercase, noise_removal, normalization, stopword_removal, lemmatization
tasks = ['lowercase', 'noise_removal', 'normalization', 'lemmatization']
preprocessing_pipeline = Pipeline([
    ('preprocesing_text', PreprocessingText(tasks=tasks, keep_tags=False)),
    ('vect', CountVectorizer(ngram_range=(1,1))),
    # ('tfidf', TfidfTransformer())
    # Pq tfidf reduire l'efficacite : https://stackoverflow.com/questions/39152229/in-general-when-does-tf-idf-reduce-accuracy:w
])

# Il va sauvegarde tous les tokens du X_train
# Quand on transforme X_valid, tous les tokens qui ne sont pas reconnus dans X_train vont etre disparus
preprocessing_pipeline.fit(X_train)
X_train = preprocessing_pipeline.transform(X_train).toarray()
X_valid = preprocessing_pipeline.transform(X_valid).toarray()
print(X_valid.shape)

#%% Train
model, losses, f1_valid = train(X_train, y_train, X_valid, y_valid, epochs=200)
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