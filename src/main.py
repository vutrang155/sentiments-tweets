from src.preprocessing import ReadTweet, PreprocessingText
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# A modifier, ça dépend de votre workdir, pour moi c'est 'sentiments-tweets'. Si vous êtes
# à 'sentiments-tweets/src' => '../data/eit_annot_100_1899.txt"
FILEPATH = "data/eit_annot_100_1899.txt"

# Get tweets
X, y = ReadTweet(FILEPATH).get_data()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, test_size=0.3, random_state=43)

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
X_train = preprocessing_pipeline.fit_transform(X_train)
X_valid = preprocessing_pipeline.fit_transform(X_valid)