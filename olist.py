import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from dotenv import load_dotenv
from plotly.offline import iplot
from pywaffle import Waffle
from sklearn.cluster import KMeans

load_dotenv()


# Rappel convention PEP8 : https://visualgit.readthedocs.io/en/latest/pages/naming_convention.html


def assign_frequency(frequency):
    """
    Fonction permettant d'attribuer un score de fréquence

    Entrée :
    - fréquence - commande passée, int

    Sortie :
    - F - score de fréquence

    """

    if frequency >= 7:
        return 4
    elif frequency >= 4:
        return 3
    elif frequency >= 2:
        return 2
    else:
        return 1


def convert_to_dt(dataframe, columns, dt_format=None):
    """
    Fonction prenant en compte le nom du dataframe et les colonnes de date pour la conversion au format date

    Entrée :
    - Dataframe

    Sortie :
    - None (Convertit le format de la colonne en datetime)

    """
    for column in columns:
        dataframe[column] = pd.to_datetime(dataframe[column],
                                           format=dt_format).dt.date


def k_means_func(dataframe, n_clusters):
    """
    Fonction permettant de calculer la somme des erreurs quadratiques pour un nombre donné de clusters.

    Entrées :
    - dataframe - dataframe avec des données normalisées
    - n_clusters - nombre de clusters

    Sortie :
    - sse - somme des erreurs quadratiques

    """
    k_means = KMeans(n_clusters=n_clusters, random_state=1)
    k_means.fit(dataframe)

    return k_means.inertia_


def plot_map(dataframe,
             title,
             lower_bound,
             upper_bound,
             metric,
             maker_size=3,
             is_sub_segment=False):
    """
    Fonction de visualisation de données géographiques de métriques démographiques.

    Entrée :
    - dataframe - dataframe avec la feature target
        (métrique ; champ de code couleur nécessaire si sous_segment à visualiser)
    - title - texte à afficher comme titre du graphique
    - lower_bound - seuil inférieur de l'échelle de couleurs
    - upper_bound - seuil supérieur de l'échelle de couleurs
    - metric - caractéristique/ métrique kpi à visualiser
    - is_sub_segment - booléen,
        si "True" : le sous-segment sera visualisé avec un code couleur,
        si "False", valeur conforme à la couleur
    - marker_size - taille du marqueur

    Sortie :
    - Visualisation des données géographiques

    """

    if is_sub_segment is True:
        dict_marker = dict(
            size=maker_size,
            color=dataframe.color,
        )
    else:
        dict_marker = dict(size=maker_size,
                           color=dataframe[metric],
                           showscale=True,
                           colorscale=[[0, 'blue'], [1, 'red']],
                           cmin=lower_bound,
                           cmax=upper_bound)

    data_geo = [
        go.Scattermapbox(lon=dataframe['geolocation_lng'],
                         lat=dataframe['geolocation_lat'],
                         marker=dict_marker)
    ]

    layout = dict(title=title,
                  showlegend=False,
                  mapbox=dict(
                      accesstoken=os.getenv("MAPBOX_TOKEN"),
                      center=dict(lat=-23.5, lon=-46.6),
                      bearing=10,
                      pitch=0,
                      zoom=2,
                  ))
    fig = dict(data=data_geo, layout=layout)
    iplot(fig, validate=False)


def plot_waffle_chart(dataframe, metric, agg, title_txt, group='sub_segment'):
    """
    Fonction permettant de créer un graphique en forme de gaufre.
    La visualisation montre comment les sous-segments de clients sont répartis selon des métriques définies.

    Entrée :
    - dataframe
    - métrique - feature/ métrique kpi à visualiser
    - agg - méthode d'agrégation
    - title_txt - texte à afficher comme titre du graphique

    Sortie :
    - Un délicieux graphique gaufré

    """
    data_revenue = dict(
        round(dataframe.groupby(group).agg({metric: agg}))[metric])

    plt.figure(FigureClass=Waffle,
               rows=5,
               columns=10,
               values=data_revenue,
               labels=[f"{k, v}" for k, v in data_revenue.items()],
               legend={
                   'loc': 'lower left',
                   'bbox_to_anchor': (1, 0)
               },
               figsize=(8, 5))

    plt.title(title_txt)


def rfm_assiner(dataframe):
    """
    TODO : Rendre la fonction flexible

    Fonction permettant d'attribuer des classes RFM selon les conditions.

    Entrée :
    - dataframe - dataframe contenant le score RFM et la clé du segment RFM.

    Sortie :
    - renvoie une classe de segment RFM (str)

    """
    if (int(dataframe['segment_RFM']) >= 434) or (dataframe['score_rfm'] >= 9):
        return 'Meilleur client'
    elif (dataframe['score_rfm'] >= 8) and (dataframe['M'] == 4):
        return 'Dépensier'
    elif (dataframe['score_rfm'] >= 6) and (dataframe['F'] >= 2):
        return 'Fidèle'
    elif (int(dataframe['segment_RFM']) >= 231) or (dataframe['score_rfm'] >= 6):
        return 'Fidélité potentielle'
    elif ((int(dataframe['segment_RFM']) >= 121) and
          (dataframe['R'] == 1)) or dataframe['score_rfm'] == 5:
        return 'Presque perdu'
    elif (dataframe['score_rfm'] >= 4) and (dataframe['R'] == 1):
        return 'En hibernation'
    else:
        return 'Client perdu'


def rfm_iso_scatter(dataframe, x, y, z):
    """
    Fonction permettant de générer un nuage de points en 3D

    Entrée :
    - dataframe
    - x, y, z - 3 features pour les trois coordonnées

    Sortie :
    - Aucun (nuage de points 3D interactif)
    """
    x = dataframe[x]
    y = dataframe[dataframe[y] < 5][y]
    z = dataframe[dataframe[z] < 4000][z]

    fig = go.Figure(data=[
        go.Scatter3d(x=x,
                     y=y,
                     z=z,
                     mode='markers',
                     marker=dict(
                         size=1, color=y, colorscale='thermal', opacity=0.8))
    ])

    fig.update_layout(scene=dict(xaxis_title='Récence',
                                 yaxis_title='Fréquence',
                                 zaxis_title='Montant'),
                      width=700,
                      margin=dict(r=20, b=10, l=10, t=10))

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def subst_mean(dataframe, columns):
    """
    La fonction prend le nom du dataframe et la liste des colonnes à substituer,
    les NaN seront remplies avec la moyenne.

    Entrée :
    - Dataframe

    Sortie :
    - None (Convertit le format de la colonne en datetime)

    """
    for column in columns:
        dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
