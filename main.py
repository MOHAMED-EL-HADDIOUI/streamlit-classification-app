import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

st.title('Application de Classification Avancée')

st.write("""
# Explorer différents classifieurs et jeux de données
Analysez les performances et visualisez les résultats!
""")

dataset_name = st.sidebar.selectbox(
    'Sélectionnez le jeu de données',
    ('Iris', 'Cancer du sein', 'Vin')
)

st.write(f"## Jeu de données : {dataset_name}")

classifier_name = st.sidebar.selectbox(
    'Sélectionnez le classifieur',
    ('KNN', 'SVM', 'Random Forest')
)


def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Vin':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y, data.feature_names


X, y, feature_names = get_dataset(dataset_name)
st.write('Forme du jeu de données:', X.shape)

# Option pour déterminer manuellement le nombre de classes
manual_classes = st.sidebar.checkbox('Définir manuellement le nombre de classes')
if manual_classes:
    num_classes = st.sidebar.number_input('Nombre de classes', min_value=2, max_value=10, value=len(np.unique(y)))
else:
    num_classes = len(np.unique(y))

st.write('Nombre de classes:', num_classes)

# Affichage des statistiques descriptives
df = pd.DataFrame(X, columns=feature_names)
st.write("### Statistiques descriptives")
st.write(df.describe())

# Matrice de corrélation
st.write("### Matrice de corrélation")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        kernel = st.sidebar.selectbox('Kernel', ('rbf', 'linear'))
        params['C'] = C
        params['kernel'] = kernel
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'], kernel=params['kernel'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'], random_state=1234)
    return clf


clf = get_classifier(classifier_name, params)

#### CLASSIFICATION ####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)

st.write(f'### Résultats pour le classifieur : {classifier_name}')
st.write(f'Précision : {acc:.2f}')

# Matrice de confusion
st.write("### Matrice de confusion")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
st.pyplot(fig)

# Rapport de classification
st.write("### Rapport de classification")
clf_report = classification_report(y_test, y_pred, output_dict=True)
st.table(pd.DataFrame(clf_report).transpose())

#### TRACÉ DU JEU DE DONNÉES ####
st.write("### Visualisation des données (PCA)")
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.colorbar(scatter)

st.pyplot(fig)

# Importance des caractéristiques (pour Random Forest)
if classifier_name == 'Random Forest':
    st.write("### Importance des caractéristiques")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title("Importance des caractéristiques")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation='vertical')
    plt.tight_layout()
    st.pyplot(fig)

# Distribution des caractéristiques
st.write("### Distribution des caractéristiques")
feature_to_plot = st.selectbox("Choisissez une caractéristique à visualiser", options=feature_names)
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x=feature_to_plot, hue=y, kde=True, ax=ax)
plt.title(f"Distribution de {feature_to_plot}")
st.pyplot(fig)