#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 10:20:37 2021

@author: david
"""

import codecs
import pickle
from text_utils import transform_to_caps

filepath = "liste.de.mots.francais.frgut.txt"

whole_dico = ""
with codecs.open(filepath, "r", "utf-8") as lines:
    for l in  lines:
        whole_dico += l[:-1] + " "
        
dico = transform_to_caps(whole_dico)
# We add simple letters and qu'
words = ['QU'] + [chr(i) for i in range(65,91)] + dico.split(" ")
    
pickle.dump(words, open( "dictionnary.data", "wb" ))