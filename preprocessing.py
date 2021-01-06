import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# ReadTweet
class ReadTweet :
    '''
    Permet de récuperer les tweets d'un auteur depuis un fichier
    '''
    def __init__(self, filepath, author="consensus"):
        '''
        Initialiser les TweetReader, pour récupérer les tweets, utiliser la méthode get_data
        :param filepath: lien vers le fichier de données
        :param author: l'auteur des tweets à exporter, "consensus" par défaut
        '''
        with open(filepath, "r") as f:
            content = f.readlines()

        info = [x.split(")")[0] for x in content]

        # Get only the line where the author = author
        idx_consensus = np.array([y.split(",")[2] for y in info])
        idx_consensus = np.where(idx_consensus == author)

        self.results = np.array([y.split(",")[1] for y in info])[idx_consensus]

        # tweets
        self.tweets = np.array([x.split(")")[1] for x in content])[idx_consensus]


    def get_data(self):
        '''
        Récuperer les données exportées
        :return: tweets et leurs sentiments
        '''
        return self.tweets, self.results

class Preprocessing :
    '''
    Préprocessing des données
    '''
    DEFAULT_TASKS = ["lowercase", "noise_removal", "normalization", "stopword_removal", "lemmatization"]
    MODELS = ["bow", "tf-idf"]

    def __init__(self, X, tasks=DEFAULT_TASKS, model="bow"):
        '''
        Initialise la préprocessing
        :param X: Tweets à préprocessing
        '''

        self.X = X
        ########## TASKS ##############
        ###############################
        # MUST
        ########################
        # Lowercase :
        if "lowercase" in tasks:
            self.lowercase()
        # Noise Removal :
        if "noise_removal" in tasks:
            self.noise_removal()
        #########################

        # SHOULD
        #########################
        # Normalization
        if "normalization" in tasks:
            self.normalization()
        #########################

        # DEPEND
        #########################
        # Stop-word Removal
        if "stopword_removal" in tasks:
            self.stopword_removal()
        # Lemmatization
        if "lemmatization" in tasks:
            self.lemmatization()
        #########################
        ########## MODEL ##############
        ###############################
        if model == "bow":
            pass
        elif model == "tf-idf":
            pass

    def lowercase(self):
        '''
        Rendre les tweets en miniscule
        '''
        # TODO
        pass

    def noise_removal(self):
        '''
        Enlever les caractères spéciaux collés à un token
        '''
        # TODO
        pass

    def normalization(self):
        '''
        Normaliser les textes (":)" -> "smiley")
        '''
        # TODO
        pass

    def stopword_removal(self):
        '''
        Supprimer les stop-words
        '''
        # TODO
        pass

    def lemmatization(self):
        '''
        Mettre les mots en forme initative
        '''
        # TODO
        pass

    def get(self):
        '''
        Renvoie les données traitées
        :return: X
        '''
        return self.X