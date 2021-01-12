import numpy as np
import re

from utils import string_utils
import constants

from sklearn.base import BaseEstimator, TransformerMixin

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

class PreprocessingText (BaseEstimator, TransformerMixin):
    """
    Préprocessing des données
    """
    DEFAULT_TASKS = ["lowercase", "noise_removal", "normalization", "stopword_removal", "lemmatization"]

    def __init__(self, tasks=DEFAULT_TASKS, keep_tags=True):
        '''
        Initialiser la préprocessing
        :param tasks: list, Les tachês à passer. Par défaults, elles sont définies dans le variable constant DEFAULT_TASKS
        :param keep_tags: boolean, True si l'on veut garder les hashtags et les mentions tags
        '''
        self.keep_tags = keep_tags
        self.tasks = tasks

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        ########## TASKS ##############
        ###############################
        # MUST
        ########################
        if "lowercase" in self.tasks:
            X = self.lowercase(X)
        #########################
        if "normalization" in self.tasks:
            X = self.normalization(X)
        #########################
        if "noise_removal" in self.tasks:
            X = self.noise_removal(X)
        #########################


        # DEPEND
        #########################
        if "stopword_removal" in self.tasks:
            X = self.stopword_removal(X)
        if "lemmatization" in self.tasks:
            X = self.lemmatization(X)
        #########################
        return X

    def lowercase(self, X):
        '''
            Rendre les tweets en miniscule
        :param X: ndarray de dimension (n,)
        :return: ndarray de dimension (n,) : tweets en miniscule
        '''
        return np.char.lower(X)

    def normalization(self, X):
        '''
        Normaliser les textes ("http://..." -> "##WEBSITE##")
        Si deux emojis sont collés, on les séparent par une espace
        :param X: ndarray (n, )
        :return: ndarray (n,) : tweets normalisés
        '''
        for i in range(X.shape[0]):
            X[i] = re.sub(r'http\S+', '##WEBSITE##', X[i])
        return X

    def noise_removal(self, X):
        '''
            Enlever les caractères spéciaux collés à un token .I.E. ":)hello" => ":) hello"
        :param X: ndarray de dimension (n,)
        :return: ndarray de dimension (n,) : Tweets où les bruits sont enlevés
        '''
        for i in range(X.shape[0]):
            X[i] = string_utils.decontract(X[i])

            # TODO: ":)hello" to ":) hello"
            #   N'oublie pas keep_hashtags, keep_mention_tags.
            #   Si keep_hashtags = False => :)#Hello = :) # Hello
            #   Si non                   => :)#Hello = :) #Hello
            prefix = ""
            if self.keep_tags == True:
                prefix = "@#"
            # ATTENTION : Ce code n'a pas marché pour les langues étrangères
            # Ceci ne marche que sur le code ASCII
            #l = re.findall(r"[A-Za-z@#]+|\S", X[i])
            # Resolution :
            new_str = ""
            for w in X[i].split():
                # Decomposer en lsite
                l = re.findall(r"[_" + prefix + constants.REGEX_UNICODE_WORD + r"]+|\S", w) # Exemple : "Hello:3#Mynameis" renvoie ["Hello" ":", "3", "#Mynameis"]

                # Rematch en string
                # ["Hello" ":", "3", "#Myanmeis"] => "Hello :3 #Mynameis"
                new_substr = ""
                for element in l:
                    # Check if element in list is a word (or hashtag, ...)
                    if bool(re.match(r"[_" + prefix + constants.REGEX_UNICODE_WORD + r"]+", element)):
                        new_substr = new_substr + " " + element + " "
                    else :
                        new_substr = new_substr + element

                # Replace w by new_str
                new_str = new_str+new_substr
            X[i] = new_str
        return X


    def stopword_removal(self, X):
        '''
        Supprimer les stop-words
        :param X: ndarray (n, )
        :return: ndarray (n,) : tweets ou les stopwords sont supprimer
        '''
        # TODO
        return X

    def lemmatization(self, X):
        '''
        Mettre les mots en forme infinitive
        :param X: ndarray (n, )
        :return: ndarray (n,) : tweets en forme infinitive
        '''
        # TODO
        return X