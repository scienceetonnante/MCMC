#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from text_utils import load_corpus, transform_to_caps, count_bigrams, char_to_id, id_to_char, count_correct_words, score_correct_words, frequency_order, apply_code, invert_code
import pickle

# =============================================================================

# CREATE BRIGRAMS (if needed)
COUNT_BIGRAMS = True

corpus_filename = "swann.txt"
bigrams_filename = "bigrams.dat"

if COUNT_BIGRAMS:
    corpus = load_corpus(corpus_filename)
    corpus = transform_to_caps(corpus)    
    count_bigrams(corpus, bigrams_filename, "bigrams.png")

# =============================================================================
    
# LOAD BIGRAMS and build log(probability) matrix
    
bigrams = np.fromfile(bigrams_filename,dtype="int32").reshape(27,27)
p=bigrams.astype('float')/np.tile(sum(bigrams.T),(27,1)).T
p[np.isnan(p)] = 0
EPSILON = 1e-6
logp = np.log(p + EPSILON)

# =============================================================================

# Log likelihood per caracter
def likelihood(s):
    res = 0
    c1 = s[0]
    for c2 in s[1:]:
        i = char_to_id(c1)
        j = char_to_id(c2)
        res += logp[i,j]
        c1 = c2
        
    return res/len(s)



# Apply one permutation to a code
def permute_code(code,i,j):
    newcode = code.copy()
    newcode[j] = code[i]
    newcode[i] = code[j]
    return newcode


# Text to compute frequency of letter for initial guess
freq_text = transform_to_caps(load_corpus("swann.txt"))

ciphered_text = "Ci kmvjpx vb ci wmoxi"

# =============================================================================
# =============================================================================

# Text to decipher : choose among the following options

# =============================================================================

## 1) TEST TEXT : random sentence from corpus
  
#test_text = transform_to_caps(load_corpus("swann.txt"))
#START = np.random.randint(180000)
#SIZE = 30 # Nb de mots
#original = " ".join(test_text.split(" ")[START:(START+SIZE)])
#true_code = [0] + list(np.random.permutation(range(1,27)))
#ciphered_text = apply_code(original, true_code)

# =============================================================================

# 2) FOR ERIC


#foreric = """NASJX OXH NXH SOE BXYA SJXEA PY SNXYZEA ! YH ZASJSEG BSP ZAEJESG, RSA G SP ZY JY ? OS BAXBXPEZEXH EHRGYZ GS DEPBSAEZEXH, P'SKKASHRUEPPSHZ D'YH SZZAENYZ BSAOE JEHTZ-PEM. BSP RXH ! (XYE C'SE KSEZ KXAZ) 
#           JXEGS QYE PSHP PXYBRXH PSYAS ASJEA ZXH RENXYGXZ, XY DY OXEHP ZXH SAZ DY BLZUXH. SY BGSEPEA D'YH CXYA HXYP JXEA SYZXYA D'YH NXRV XY DY KGSRXH D'YH JEH, QY'EG PXEZ NGSHR XY AYNEP. ZXH JEG SOE"""
#ciphered_text = transform_to_caps(foreric)

# =============================================================================

# 3) BAER

#otis = """Mais, vous savez, moi je ne crois pas qu’il y ait de bonne ou de mauvaise situation. Moi, si je devais résumer ma vie aujourd’hui avec vous, je dirais que c’est d’abord des rencontres, des gens qui m’ont tendu la main"""#, peut-être à un moment où je ne pouvais pas, où j’étais seul chez moi. Et c’est assez curieux de se dire que les hasards, les rencontres forgent une destinée… Parce que quand on a le goût de la chose, quand on a le goût de la chose bien faite, le beau geste, parfois on ne trouve pas l’interlocuteur en face, je dirais, le miroir qui vous aide à avancer"""
#otis = """Et c’est assez curieux de se dire que les hasards, les rencontres forgent une destinée"""
#otis = """moi je ne crois pas qu’il y ait de bonne ou de mauvaise situation"""
#
#original = transform_to_caps(otis)
#np.random.seed(3)
#true_code = [0] + list(np.random.permutation(range(1,27)))
#ciphered_text = apply_code(original, true_code)
#print(" ".join([chr(64+i) for i in true_code[1:]]))
#true_decode = invert_code(true_code)
#print(" ".join([chr(64+i) for i in true_decode[1:]]))
#
#print(original + "   ("+str(len(original)) + " chars)")
#print(likelihood(original))
#print("\n\n")


# =============================================================================
# =============================================================================

ciphered_text = transform_to_caps(ciphered_text)

print(ciphered_text + " Likelihood : {0:.2f}".format(likelihood(ciphered_text)))
print("\n")

# Initial guess from frequency

ref_freq = frequency_order(freq_text)
obs_freq = frequency_order(ciphered_text)

freq_code = [0] + list(range(1,27))

for i in range(1,27):
    pos = obs_freq.index(i)
    freq_code[i] = ref_freq[pos]

cur_code = freq_code.copy()
cur_trad = apply_code(ciphered_text, cur_code)
cur_like = likelihood(cur_trad)




# Best found so far
best_code = cur_code.copy()
best_like = cur_like
best_trad = cur_trad

print(best_trad + "    N=" + str(0) + " L={0:.2f}".format(best_like))


# Now the main loop

MIN_ITER = 2000
MAX_ITER = 100000
THRESHOLD = -2.05
ALPHA = 1

for k in range(MAX_ITER):
    
    # Build a tentative move
    i = np.random.randint(1,27)
    j = np.random.randint(1,27)
    tt_code = permute_code(cur_code, i, j)
    
    tt_trad = apply_code(ciphered_text, tt_code)
    tt_like = likelihood(tt_trad)
    
    # Test whether move should be accepted    
    x = np.random.rand()
    p = np.exp(ALPHA*(tt_like - cur_like) * len(ciphered_text))
    
    if(x < p):    
        cur_code = tt_code.copy()
        cur_trad = tt_trad
        cur_like = tt_like
        #print("ACCEPT")
        
        if(cur_like > best_like):
            best_code = cur_code.copy()
            best_like = cur_like
            best_trad = cur_trad     
            print(best_trad + "    [k=" + str(k) + " L={0:.2f}]".format(best_like))
            
    if k > MIN_ITER and best_like > THRESHOLD:
        break
         
    
#######################################################################################################    

print("\nEnter second phase")      


with open('dictionnary.data', 'rb') as filehandle:
    dictionnary_words = pickle.load(filehandle)          

cnt, total = count_correct_words(best_trad, dictionnary_words) 
word_score = score_correct_words(best_trad, dictionnary_words)

print("Mots OK "+str(cnt)+"/"+str(total)+" score="+ str(word_score))     

GAMMA = 4.0
best_score = GAMMA * word_score + best_like



cur_code = best_code
cur_score = best_score
cur_trad = best_trad

NITER2 = 2000
temperature = 0.05
rho = 0.999

for k in range(NITER2):
    
    # Build a tentative move and compute score
    i = np.random.randint(1,27)
    j = np.random.randint(1,27)
    tt_code = permute_code(cur_code, i, j)    
    tt_trad = apply_code(ciphered_text, tt_code)    
    tt_word_score = score_correct_words(tt_trad, dictionnary_words)
    tt_like = likelihood(tt_trad)
    tt_score = GAMMA * tt_word_score + tt_like
    

    # Test whether move should be accepted    
    x = np.random.rand()
    p = np.exp((tt_score-cur_score)/temperature)
    temperature = temperature * rho
    
    if(x < p):        
        cur_code = tt_code.copy()
        cur_trad = tt_trad
        cur_score = tt_score
        
        if(cur_score > best_score):
            best_code = cur_code
            best_score = cur_score
            best_trad = cur_trad
            print(tt_trad + "  W={0:.2f}".format(tt_word_score))
            #print(tt_trad + "  W={0:.2f}  L={1:.2f} S={2:.2f} T={3:.3f} k={4}".format(tt_word_score, tt_like, tt_score,temperature,k))
            

