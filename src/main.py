from src.preprocessing import ReadTweet, Preprocessing

# A modifier, ça dépend de votre workdir, pour moi c'est 'sentiments-tweets'. Si vous êtes
# à 'sentiments-tweets/src' => '../data/eit_annot_100_1899.txt"
FILEPATH = "data/eit_annot_100_1899.txt"

# Get tweets
X, y = ReadTweet(FILEPATH).get_data()
n_tweets = y.shape[0]

# Il dispose un degré d'importance pour chaque tâche spécifique :
#   1. lowercase, noise_removal (priorité)
#   2. normalization
#   3. stopword_removal, lemmatization
tasks = Preprocessing.DEFAULT_TASKS # lowercase, noise_removal, normalization, stopword_removal, lemmatization
preprocessing = Preprocessing(X, tasks=tasks, model="bow")
X_lowercased = preprocessing.get()


