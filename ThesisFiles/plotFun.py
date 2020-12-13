
#CLASSE PER PLOTTARE DATASET

import pandas as pd
import numpy as np
from matrixprofile import *
from matrixprofile.discords import discords
from matplotlib import pyplot as plt
import random
from scipy.io import arff


def setVariablesForPlot(window_s):
    global window_size
    window_size=window_s


def plot_motifs(mtfs, labels, ax, data, window_size):
    #data can be raw data or MP
    colori = 0
    colors = 'rmb'
    for ms,l in zip(mtfs,labels):
        c =colors[colori % len(colors)]
        starts = list(ms)
        ends = [min(s + window_size,len(data)-1) for s in starts]
        ax.plot(starts, data[starts],  c +'o',  label=l+'(Motif)')
        ax.plot(ends, data[ends],  c +'o', markerfacecolor='none')
        for nn in ms:
            ax.plot(range(nn,nn+window_size),data[nn:nn+window_size], c , linewidth=2)
        colori += 1

    #ax.plot(a,'green', linewidth=1, label="data") COMMENTATO PERCHE PLOTTO I DATI INDIPENDENTEMENTE
    ax.legend()


def plot_discords(dis, ax, data, window_size):
    # data can be raw data or Mp
    color = 'k'
    for start in dis:
        end = start + window_size
        ax.plot(start, data[start], color, label='Discord')
        ax.plot(end, data[end], color, markerfacecolor='none')

        ax.plot(range(start, start + window_size), data[start:start + window_size], color, linewidth=2)

    ax.legend(loc=1, prop={'size': 12})


def plot_all(Ts, mp, mot, motif_dist, dis, window_size):
    # genera e compara TS con MP, motifs e discords ottenuti

    # Append np.nan to Matrix profile to enable plotting against raw data (FILL DI 0 ALLA FINE PER RENDERE LE LUNGHEZZE UGUALI )
    mp_adj = np.append(mp, np.zeros(window_size - 1) + np.nan)

    # MODO 2 PER PLOTTARE (O-ORIENTED)
    # Plot dei dati
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 15))
    ax1.plot(np.arange(len(Ts)), Ts, label="Ts",
             color='g')  # stampo linespace su x e valori data su y (USATO SE NON STAMPO MOTIF/DIS)
    ax1.set_ylabel('Ts', size=22)
    plot_motifs(mot, [f"{md:.3f}" for md in motif_dist], ax1, Ts, window_size)  # sk
    plot_discords(dis, ax1, Ts, window_size)

    # Plot della Matrix Profile
    ax2.plot(np.arange(len(mp_adj)), mp_adj, label="Matrix Profile", color='green')
    ax2.set_ylabel('Matrix Profile', size=22)
    plot_motifs(mot, [f"{md:.3f}" for md in motif_dist], ax2, mp_adj, window_size)
    plot_discords(dis, ax2, mp_adj, window_size)

    plt.show()


def retrieve_all2(Ts):  # fornita la Ts calcola e restituisce mp, motifs, motifs_distances e discords
    Ts = Ts[:len(Ts)-1]  # rimuovo l'attributo "classe"

    dfMP = pd.DataFrame(Ts).astype(float)  # genero Dframe per lavorarci su, DA CAPIRE PERCHE SERVE FLOAT
    mp, mpi = matrixProfile.stomp(dfMP[0].values, window_size)  # OK STOMP

    # PREPARO TUPLA DA PASSARE ALLA FUN MOTIF (RICHIEDE TUPLA FATTA DA MP E MPI)
    tupla = mp, mpi

    mot, motif_dist = motifs.motifs(dfMP[0].values, tupla, 2)

    # CALCOLO MOTIFS
    print('Motifs starting position: ' + str(mot) + ' Motifs values (min distances): ' + str(motif_dist))

    # CALCOLO DISCORDS
    dis = discords(mp, window_size, 2)
    print('Discords starting position: ' + str(dis))

    tupla = mp, mot, motif_dist, dis
    return tupla


# riceve la lista di coppie dei motifs per ogni record(Ts), e resittuisce lista di valori singoli

def candidateFilter2(CandidateList):
    l2 = np.array([])
    for i in range(len(CandidateList['Motif'])):  # per ogni entry (per ogni record)
        numMotif = len(CandidateList['Motif'].iloc[i])
        # print(numMotif)
        for j in range(numMotif):  # per ogni lista di motif
            l1 = CandidateList['Motif'].iloc[i]  # prima lista
            l2 = np.append(l2, l1[j][0])  # prendo primo valore di ogni lista

        CandidateList['Motif'].iloc[i] = l2
        l2 = np.array([])  # svuoto array

    return CandidateList


