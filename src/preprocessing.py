import numpy as np
import re
from utils import string_utils

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

    def __init__(self, X, tasks=DEFAULT_TASKS, model="bow", keep_tags=False):
        # TODO : making model param an object !
        '''
        Initialise la préprocessing
        :param X: Tweets à préprocessing
        '''

        self.X = X

        self.keep_tags = keep_tags
        ########## TASKS ##############
        ###############################
        # MUST
        ########################
        if "lowercase" in tasks:
            self.lowercase()
        if "noise_removal" in tasks:
            self.noise_removal()
        #########################

        # SHOULD
        #########################
        if "normalization" in tasks:
            self.normalization()
        #########################

        # DEPEND
        #########################
        if "stopword_removal" in tasks:
            self.stopword_removal()
        if "lemmatization" in tasks:
            self.lemmatization()
        #########################


        ########## MODEL ##############
        ###############################
        if model == "bow":
            # TODO
            pass
        elif model == "tf-idf":
            # TODO
            pass


    def lowercase(self):
        '''
        Rendre les tweets en miniscule
        '''
        self.X = np.char.lower(self.X)

    def noise_removal(self):
        '''
        Enlever les caractères spéciaux collés à un token .I.E. ":)hello" => ":) hello"
        '''
        for i in range(self.X.shape[0]):
            self.X[i] = string_utils.decontract(self.X[i])

            # TODO: ":)hello" to ":) hello"
            #   N'oublie pas keep_hashtags, keep_mention_tags.
            #   Si keep_hashtags = False => :)#Hello = :) # Hello
            #   Si non                   => :)#Hello = :) #Hello
            if self.keep_tags == True:
                #l = re.findall(r"[A-Za-z@#]+|\S", self.X[i])
                pass

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