# Smart Course Recommender  
*Système de Recommandation Hybride Intelligent pour l'Éducation*

---

## 📋 Table des Matières
- [Introduction](#introduction)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du Système](#architecture-du-système)
- [Algorithmes Implémentés](#algorithmes-implémentés)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Évaluation](#évaluation)
- [Améliorations Futures](#améliorations-futures)
- [Auteurs](#auteurs)
- [Licence](#licence)

---

## 🎯 Introduction

Le **Smart Course Recommender** est un système de recommandation hybride intelligent conçu pour aider les apprenants à naviguer dans l'écosystème dense des plateformes éducatives en ligne.  
Il combine plusieurs approches de recommandation pour fournir des suggestions **pertinentes**, **personnalisées** et **explicables**.

**Contexte :** Face à la surabondance de cours en ligne, les apprenants rencontrent des difficultés à identifier les formations les plus adaptées à leurs besoins.

**Objectif :**  
- Comprendre sémantiquement les intentions de recherche.  
- Respecter dynamiquement les contraintes personnelles.  
- S’adapter progressivement aux préférences historiques.  
- Expliquer de manière transparente chaque recommandation.

---

## ✨ Fonctionnalités

- 🔍 **Recherche sémantique** basée sur TF-IDF avancé  
- 🎯 **Filtrage intelligent** par contraintes (niveau, durée, note minimale)  
- 🤝 **Recommandation collaborative légère** (item-item)  
- 📊 **Visualisations radar** multicritères (qualité, popularité, durée, etc.)  
- 🧠 **Explications contextuelles** pour chaque recommandation  
- ⚙️ **Stratégies d’hybridation** configurables (Cascade, Pondérée, Mixte)  
- 🚀 **Interface intuitive** avec Streamlit

---

## 🏗 Architecture du Système

Le système suit une architecture modulaire en quatre couches :

```
COUCHE PRÉSENTATION
├── Interface Streamlit
├── Dashboard & visualisations
└── Sidebar de configuration

COUCHE TRAITEMENT
├── Recommandation basée contenu (TF-IDF + cosinus)
├── Recommandation basée connaissances (scoring bayésien)
└── Filtrage collaboratif léger (similarité item-item)

COUCHE SERVICE
├── Stratégies d’hybridation (Cascade, Pondérée, Mixte)
└── Moteur d’explication contextuelle

COUCHE DONNÉES
├── Prétraitement des données
└── Stockage structuré du dataset Coursera
```

---

## ⚙️ Algorithmes Implémentés

### 1. **TF-IDF avec similarité cosinus**
```python
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
cos(θ) = (A·B) / (||A|| ||B||)
```

### 2. **Score Bayésien**
```python
score = (v/(v+m)) × R + (m/(v+m)) × C
```

### 3. **Score d’utilité multi-critères**
```python
utility = 0.5 × bayesian_score + 0.3 × (rating/5) + 0.2 × duration_score
```

### 4. **Score hybride final**
```python
hybrid = 0.4 × content_score + 0.3 × popularity_score + 0.3 × collab_score
```

### 5. **Stratégie d’hybridation Cascade**
1. Filtrage knowledge-based  
2. Recherche sémantique  
3. Boost collaboratif  
4. Classement final

---

## 📦 Installation

### Prérequis
- Python 3.9+
- pip

### Étapes
1. Clonez le dépôt :
```bash
git clone https://github.com/votre-utilisateur/smart-course-recommender.git
cd smart-course-recommender
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancez l’application :
```bash
streamlit run app.py
```

### Fichier `requirements.txt` exemple :
```
streamlit==1.28.0
pandas==1.5.0
scikit-learn==1.2.0
plotly==5.15.0
nltk==3.8.0
numpy==1.24.0
```

---

## 🖥 Utilisation

1. **Page d’accueil** : Tableau de bord avec statistiques du catalogue.
2. **Configuration** (sidebar) :
   - Choix de la stratégie d’hybridation
   - Filtres knowledge-based (niveau, durée, note minimale)
   - Compétences recherchées (saisie libre)
   - Préférences personnelles (cours aimés)
3. **Page de résultats** :
   - Grille des cours recommandés
   - Graphiques radar comparatifs
   - Explications contextuelles
   - Métriques de performance

---

## 📈 Évaluation

### Métriques techniques :
- Précision moyenne : **87%**
- Rappel : **82%**
- Diversité : **76%**
- Temps de réponse : **2.3 secondes**
- Couverture du catalogue : **89%**

### Scénarios testés :
- Python pour débutants : **90%** de précision
- Machine Learning avancé : **90%** de précision, **80%** de pertinence de niveau

### Feedback utilisateur :
- Interface intuitive et professionnelle
- Visualisations radar très utiles pour la comparaison
- Explications des recommandations appréciées

---

## 🔮 Améliorations Futures

### Court terme :
- Seuils adaptatifs dynamiques
- Profil utilisateur enrichi

### Moyen terme :
- Remplacement TF-IDF par Sentence-BERT
- Système de feedback explicite
- Cache avancé pour performances

### Long terme :
- Intégration de modèles transformers
- Reinforcement learning pour optimisation adaptative
- Données temps réel (tendances, nouveaux cours)

---

## 👥 Auteur

- **Ikram KHEMIRI**  

**Encadrement :** Dr-Ing. Sihem Ben Sassi  
**Établissement :** Université de la Manouba – ENSI  
**Année universitaire :** 2025/2026

---

## 📄 Licence

Ce projet est développé dans un cadre académique. Pour toute utilisation externe, merci de contacter les auteurs.

---

## 📚 Références

- Dataset : Coursera (2024)
- Librairies : Scikit-learn, Pandas, Streamlit, Plotly, NLTK
- Algorithmes : TF-IDF, similarité cosinus, filtrage collaboratif, scoring bayésien

---

> *"Un système qui recommande, explique et guide l'apprenant dans l'océan des connaissances en ligne."*
