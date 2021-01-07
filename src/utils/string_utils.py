import constants
import re

def decontract(phrase, contractions_dict=constants.CONTRACTIONS_DICT):
    '''
    Déconstruire les mots tels que "i'm" en "i am"
    :param phrase: la phrase
    :param contractions_dict: le dictionnaire de constractions. le dictionnaire par défaut est défini dans le fichier
    ../constants.py
    :return: Le mot complet
    '''
    for k, v in contractions_dict.items():
        phrase = re.sub(k, v, phrase)

    return phrase