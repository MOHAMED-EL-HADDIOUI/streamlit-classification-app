# Application de Classification Avancée avec Streamlit

Cette application Streamlit permet d'explorer différents classifieurs et jeux de données, d'analyser leurs performances et de visualiser les résultats.

## Fonctionnalités

- Sélection de différents jeux de données (Iris, Cancer du sein, Vin)
- Choix entre plusieurs classifieurs (KNN, SVM, Random Forest)
- Visualisation des données avec PCA
- Affichage des statistiques descriptives
- Matrice de corrélation des caractéristiques
- Matrice de confusion pour évaluer les performances du classifieur
- Rapport de classification détaillé
- Visualisation de l'importance des caractéristiques (pour Random Forest)
- Distribution des caractéristiques

## Installation

1. Clonez ce dépôt :
   ```
   git clone https://github.com/MOHAMED-EL-HADDIOUI/streamlit-classification-app.git
   cd streamlit-classification-app
   ```

2. Créez un environnement virtuel (recommandé) :
   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
   ```

3. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

## Utilisation

Pour lancer l'application, exécutez :

```
streamlit run main.py
```

Ouvrez ensuite votre navigateur à l'adresse indiquée (généralement http://localhost:8501).

## Structure du projet

- `main.py`: Le script principal contenant l'application Streamlit
- `requirements.txt`: Liste des dépendances Python
- `README.md`: Ce fichier

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.