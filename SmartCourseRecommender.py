# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Téléchargement des stopwords NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =============================================================================
# CLASSES DU SYSTÈME DE RECOMMANDATION
# =============================================================================

class DataPreprocessor:
    """Classe de prétraitement des données"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_data(self, df):
        """Prétraitement complet du dataset"""
        df_clean = df.copy()
        
        # Nettoyage des noms de colonnes
        df_clean.columns = [col.strip().replace(' ', '_').lower() for col in df_clean.columns]
        
        # Remplissage des valeurs manquantes
        text_columns = ['course_title', 'what_you_will_learn', 'skill_gain', 'keyword', 'instructor', 'offered_by']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('').astype(str)
        
        # Nettoyage des textes
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.clean_text)
        
        # Conversion des ratings
        df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce').fillna(0)
        df_clean['number_of_review'] = pd.to_numeric(df_clean['number_of_review'], errors='coerce').fillna(0)
        
        # Nettoyage du niveau
        df_clean['level'] = df_clean['level'].fillna('Not Specified').astype(str)
        
        # Extraction de la durée en semaines
        df_clean['duration_weeks'] = df_clean['duration_to_complete_(approx.)'].apply(self.extract_duration_weeks)
        
        # Création de tags combinés pour la recherche sémantique
        df_clean['combined_tags'] = (
            df_clean['course_title'] + " " + 
            df_clean['what_you_will_learn'] + " " + 
            df_clean['skill_gain'] + " " + 
            df_clean['keyword'] + " " +
            df_clean['instructor'] + " " +
            df_clean['offered_by']
        )
        
        # S'assurer qu'il n'y a pas de NaN dans combined_tags
        df_clean['combined_tags'] = df_clean['combined_tags'].fillna('').astype(str)
        
        # Filtrer les documents vides
        df_clean = df_clean[df_clean['combined_tags'].str.strip() != '']
        
        return df_clean
    
    def clean_text(self, text):
        """Nettoie et normalise le texte"""
        if pd.isna(text) or text == 'Not specified' or text == 'nan':
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Supprime la ponctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Supprime les espaces multiples
        return text
    
    def extract_duration_weeks(self, duration_str):
        """Extrait la durée en semaines"""
        if pd.isna(duration_str):
            return 8
        
        try:
            duration_str = str(duration_str).lower()
            
            if 'hour' in duration_str:
                hours = int(''.join(filter(str.isdigit, duration_str.split()[0])))
                return max(1, hours // 10)  # Approximation: 10 heures = 1 semaine
            elif 'week' in duration_str:
                return int(''.join(filter(str.isdigit, duration_str.split()[0])))
            elif 'month' in duration_str:
                months = int(''.join(filter(str.isdigit, duration_str.split()[0])))
                return months * 4
            else:
                # Si c'est un nombre, suppose que c'est en semaines
                return float(duration_str)
        except:
            return 8  # Durée par défaut

class ContentBasedRecommender:
    """Système de recommandation basé sur le contenu"""
    
    def __init__(self, df):
        self.df = df
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._build_tfidf_model()
    
    def _build_tfidf_model(self):
        """Construit le modèle TF-IDF"""
        # S'assurer que tous les documents sont des strings non vides
        documents = self.df['combined_tags'].fillna('').astype(str)
        valid_docs = documents[documents.str.strip() != '']
        
        if len(valid_docs) == 0:
            return
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_docs)
        except Exception as e:
            st.error(f"❌ Erreur lors de la construction du modèle TF-IDF: {e}")
    
    def semantic_search(self, query, top_n=10):
        """Recherche sémantique basée sur la requête"""
        if not query or query.strip() == "" or self.tfidf_vectorizer is None:
            return self.df.head(0)
        
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Seuil de similarité minimum
            threshold = 0.05
            similar_indices = np.where(similarities >= threshold)[0]
            
            if len(similar_indices) == 0:
                return self.df.head(0)
            
            # Tri par similarité
            results_df = self.df.iloc[similar_indices].copy()
            results_df['similarity_score'] = similarities[similar_indices]
            results_df = results_df.sort_values('similarity_score', ascending=False).head(top_n)
            
            return results_df
        except Exception as e:
            return self.df.head(0)
    
    def find_similar_courses(self, course_title, top_n=8):
        """Trouve des cours similaires à un cours donné"""
        try:
            if self.tfidf_matrix is None:
                return self.df.head(0)
                
            course_idx = self.df[self.df['course_title'] == course_title].index[0]
            similarities = cosine_similarity(
                self.tfidf_matrix[course_idx], 
                self.tfidf_matrix
            ).flatten()
            
            # Exclure le cours lui-même
            similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
            results_df = self.df.iloc[similar_indices].copy()
            results_df['similarity_score'] = similarities[similar_indices]
            
            return results_df
        except:
            return self.df.head(0)

class KnowledgeBasedRecommender:
    """Système de recommandation basé sur les connaissances"""
    
    def __init__(self, df):
        self.df = df
        self._calculate_utility_scores()
    
    def _calculate_utility_scores(self):
        """Calcule les scores d'utilité pour tous les cours"""
        # Score de popularité bayésien
        C = self.df['rating'].mean() if len(self.df) > 0 else 3.0
        m = self.df['number_of_review'].quantile(0.6) if len(self.df) > 0 else 10
        v = self.df['number_of_review']
        R = self.df['rating']
        
        self.df['bayesian_score'] = (v / (v + m)) * R + (m / (v + m)) * C
        
        # Score de durée (plus court = mieux)
        max_duration = self.df['duration_weeks'].max() if len(self.df) > 0 else 20
        self.df['duration_score'] = 1 - (self.df['duration_weeks'] / max_duration)
        
        # Score d'utilité global
        self.df['utility_score'] = (
            0.5 * self.df['bayesian_score'] +
            0.3 * (self.df['rating'] / 5.0) +
            0.2 * self.df['duration_score']
        )
    
    def constraint_based_filter(self, constraints):
        """Filtrage basé sur les contraintes"""
        candidates = self.df.copy()
        
        # Niveau
        if constraints.get('level') and constraints['level'] != 'All':
            candidates = candidates[candidates['level'] == constraints['level']]
        
        # Note minimale
        if constraints.get('min_rating'):
            candidates = candidates[candidates['rating'] >= constraints['min_rating']]
        
        # Durée maximale
        if constraints.get('max_duration_weeks'):
            candidates = candidates[
                candidates['duration_weeks'] <= constraints['max_duration_weeks']
            ]
        
        # Compétences recherchées
        if constraints.get('required_skills'):
            skill_filter = candidates['skill_gain'].apply(
                lambda x: any(skill.lower() in str(x).lower() 
                            for skill in constraints['required_skills'])
            )
            candidates = candidates[skill_filter]
        
        # Organisation
        if constraints.get('offered_by'):
            candidates = candidates[
                candidates['offered_by'].str.contains(
                    constraints['offered_by'], case=False, na=False
                )
            ]
        
        return candidates
    
    def get_trending_courses(self, top_n=10):
        """Retourne les cours tendances"""
        if len(self.df) == 0:
            return self.df
        return self.df.nlargest(top_n, 'utility_score')

class CollaborativeLightRecommender:
    """Recommandation collaborative légère (item-item)"""
    
    def __init__(self, tfidf_matrix, df):
        self.similarity_matrix = cosine_similarity(tfidf_matrix) if tfidf_matrix is not None else None
        self.df = df
    
    def item_item_recommendations(self, liked_courses, top_n=10):
        """Recommandations basées sur des cours aimés"""
        if not liked_courses or self.similarity_matrix is None:
            return pd.DataFrame()
        
        all_similarities = []
        
        for course_title in liked_courses:
            try:
                course_idx = self.df[self.df['course_title'] == course_title].index[0]
                similarities = list(enumerate(self.similarity_matrix[course_idx]))
                all_similarities.extend(similarities)
            except:
                continue
        
        if not all_similarities:
            return pd.DataFrame()
        
        # Agrégation des similarités
        similarity_scores = {}
        for idx, score in all_similarities:
            if idx not in similarity_scores:
                similarity_scores[idx] = 0
            similarity_scores[idx] += score
        
        # Exclusion des cours déjà aimés
        liked_indices = []
        for course in liked_courses:
            try:
                liked_indices.append(self.df[self.df['course_title'] == course].index[0])
            except:
                continue
        
        for idx in liked_indices:
            if idx in similarity_scores:
                del similarity_scores[idx]
        
        # Sélection des meilleurs
        if not similarity_scores:
            return pd.DataFrame()
            
        top_indices = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        results_df = self.df.iloc[[idx for idx, score in top_indices]].copy()
        results_df['collaborative_score'] = [score for idx, score in top_indices]
        
        return results_df

class HybridCourseRecommender:
    """Système de recommandation hybride principal"""
    
    def __init__(self, df):
        self.df = df
        self.preprocessor = DataPreprocessor()
        self.df_clean = self.preprocessor.preprocess_data(df)
        
        # Vérifier que le dataset n'est pas vide après nettoyage
        if len(self.df_clean) == 0:
            return
        
        # Initialisation des recommandeurs
        self.content_recommender = ContentBasedRecommender(self.df_clean)
        self.knowledge_recommender = KnowledgeBasedRecommender(self.df_clean)
        self.collaborative_recommender = CollaborativeLightRecommender(
            self.content_recommender.tfidf_matrix, self.df_clean
        )
    
    def hybrid_recommend(self, user_input):
        """
        Génère des recommandations hybrides
        """
        # Vérifier que le système est initialisé
        if len(self.df_clean) == 0:
            return pd.DataFrame()
        
        strategy = user_input.get('strategy', 'cascade')
        
        if strategy == 'cascade':
            return self._cascade_hybrid(user_input)
        elif strategy == 'weighted':
            return self._weighted_hybrid(user_input)
        elif strategy == 'mixed':
            return self._mixed_hybrid(user_input)
        else:
            return self._cascade_hybrid(user_input)
    
    def _cascade_hybrid(self, user_input):
        """Stratégie cascade : filtrage progressif"""
        # Étape 1: Filtrage knowledge-based
        filtered_courses = self.knowledge_recommender.constraint_based_filter(
            user_input.get('filters', {})
        )
        
        # Étape 2: Recherche content-based
        content_results = pd.DataFrame()
        if user_input.get('search_query'):
            content_results = self.content_recommender.semantic_search(
                user_input['search_query'], 
                top_n=20
            )
        
        # Fusion des résultats
        if not filtered_courses.empty and not content_results.empty:
            all_candidates = pd.concat([filtered_courses, content_results]).drop_duplicates()
        elif not filtered_courses.empty:
            all_candidates = filtered_courses
        else:
            all_candidates = content_results
        
        # Étape 3: Boost collaboratif
        collaborative_results = pd.DataFrame()
        if user_input.get('liked_courses') and not all_candidates.empty:
            collaborative_results = self.collaborative_recommender.item_item_recommendations(
                user_input['liked_courses'], top_n=15
            )
        
        # Fusion finale
        if not collaborative_results.empty:
            final_candidates = pd.concat([all_candidates, collaborative_results]).drop_duplicates()
        else:
            final_candidates = all_candidates
        
        # Étape 4: Classement hybride
        if not final_candidates.empty:
            final_ranking = self._hybrid_ranking(final_candidates, user_input)
            return final_ranking.head(15)
        else:
            # Fallback: cours tendances
            return self.knowledge_recommender.get_trending_courses(10)
    
    def _weighted_hybrid(self, user_input):
        """Stratégie pondérée : combinaison linéaire des scores"""
        return self._cascade_hybrid(user_input)
    
    def _mixed_hybrid(self, user_input):
        """Stratégie mixte : résultats séparés par type"""
        recommendations = {}
        
        # Content-based
        if user_input.get('search_query'):
            recommendations['content_based'] = self.content_recommender.semantic_search(
                user_input['search_query'], top_n=5
            )
        
        # Knowledge-based
        recommendations['knowledge_based'] = self.knowledge_recommender.constraint_based_filter(
            user_input.get('filters', {})
        ).head(5)
        
        # Collaborative
        if user_input.get('liked_courses'):
            recommendations['collaborative'] = self.collaborative_recommender.item_item_recommendations(
                user_input['liked_courses'], top_n=5
            )
        
        # Trending
        recommendations['trending'] = self.knowledge_recommender.get_trending_courses(5)
        
        return recommendations
    
    def _hybrid_ranking(self, candidates, user_input):
        """Classement final avec scores hybrides"""
        candidates = candidates.copy()
        
        # Score de contenu
        if user_input.get('search_query'):
            content_scores = []
            for idx, course in candidates.iterrows():
                try:
                    course_idx = self.df_clean[self.df_clean['course_title'] == course['course_title']].index[0]
                    similarity = self.content_recommender.tfidf_matrix[course_idx]
                    query_vec = self.content_recommender.tfidf_vectorizer.transform([user_input['search_query']])
                    content_score = cosine_similarity(query_vec, similarity).flatten()[0]
                    content_scores.append(content_score)
                except:
                    content_scores.append(0.3)
            candidates['content_score'] = content_scores
        else:
            candidates['content_score'] = 0.3
        
        # Score collaboratif
        if user_input.get('liked_courses'):
            collab_scores = []
            for idx, course in candidates.iterrows():
                try:
                    course_idx = self.df_clean[self.df_clean['course_title'] == course['course_title']].index[0]
                    total_similarity = 0
                    count = 0
                    for liked_course in user_input['liked_courses']:
                        try:
                            liked_idx = self.df_clean[self.df_clean['course_title'] == liked_course].index[0]
                            similarity = self.collaborative_recommender.similarity_matrix[course_idx, liked_idx]
                            total_similarity += similarity
                            count += 1
                        except:
                            continue
                    collab_scores.append(total_similarity / count if count > 0 else 0.1)
                except:
                    collab_scores.append(0.1)
            candidates['collab_score'] = collab_scores
        else:
            candidates['collab_score'] = 0.2
        
        # Score de popularité
        candidates['popularity_score'] = candidates['utility_score']
        
        # Score hybride final
        candidates['hybrid_score'] = (
            0.4 * candidates['content_score'] +
            0.3 * candidates['popularity_score'] +
            0.3 * candidates['collab_score']
        )
        
        return candidates.sort_values('hybrid_score', ascending=False)

# =============================================================================
# FONCTIONS UTILITAIRES ET VISUALISATIONS
# =============================================================================

def generate_explanation(course, user_input):
    """Génère une explication personnalisée pour la recommandation"""
    explanations = []
    
    if user_input.get('search_query'):
        explanations.append(f"🔍 **Correspond à votre recherche :** \"{user_input['search_query']}\"")
    
    filters = user_input.get('filters', {})
    if filters.get('level') and filters['level'] != 'All':
        explanations.append(f"🎯 **Niveau adapté :** {filters['level']}")
    
    if filters.get('min_rating'):
        explanations.append(f"⭐ **Dépasse la note minimale :** {filters['min_rating']}+")
    
    if user_input.get('liked_courses'):
        explanations.append("📚 **Lié aux cours que vous avez appréciés**")
    
    # Explications basées sur les métriques du cours
    if course['rating'] >= 4.5:
        explanations.append("🌟 **Excellente notation communautaire**")
    elif course['rating'] >= 4.0:
        explanations.append("👍 **Très bien noté par les apprenants**")
    
    if course['number_of_review'] > 1000:
        explanations.append("📊 **Populaire avec de nombreux avis**")
    
    if course['duration_weeks'] <= 4:
        explanations.append("⚡ **Formation intensive et courte**")
    elif course['duration_weeks'] <= 8:
        explanations.append("📅 **Durée modérée bien équilibrée**")
    
    if len(explanations) == 0:
        explanations.append("🎉 **Découverte optimisée par notre intelligence artificielle**")
    
    return "\n\n".join(explanations)

def create_radar_chart(course, max_values, course_index):
    """Crée un graphique radar pour visualiser les caractéristiques du cours"""
    categories = ['Qualité', 'Popularité', 'Intensité', 'Durée', 'Pertinence']
    
    # Normalisation des valeurs
    quality = (course['rating'] / 5.0) * 100
    popularity = min(100, (course['number_of_review'] / max(1, max_values['max_reviews'])) * 100)
    intensity = 100 - (course['duration_weeks'] / max(1, max_values['max_duration'])) * 100
    duration_score = max(20, 100 - (course['duration_weeks'] / max(1, max_values['max_duration'])) * 80)
    relevance = min(100, course.get('hybrid_score', 0.5) * 100)
    
    values = [quality, popularity, intensity, duration_score, relevance]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],  # Fermer le radar
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(100, 150, 255, 0.3)',
        line=dict(color='rgb(100, 150, 255)'),
        name=course['course_title'][:30] + "..."
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        title=f"Profil du Cours #{course_index + 1}"
    )
    
    return fig




# Ajoutez cette fonction utilitaire pour créer des graphiques radar avec des clés uniques
def display_radar_chart(course, max_values, chart_key):
    """Affiche un graphique radar avec une clé unique"""
    categories = ['Qualité', 'Popularité', 'Intensité', 'Durée', 'Pertinence']
    
    # Normalisation des valeurs
    quality = (course['rating'] / 5.0) * 100
    popularity = min(100, (course['number_of_review'] / max(1, max_values['max_reviews'])) * 100)
    intensity = 100 - (course['duration_weeks'] / max(1, max_values['max_duration'])) * 100
    duration_score = max(20, 100 - (course['duration_weeks'] / max(1, max_values['max_duration'])) * 80)
    relevance = min(100, course.get('hybrid_score', 0.5) * 100)
    
    values = [quality, popularity, intensity, duration_score, relevance]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(100, 150, 255, 0.3)',
        line=dict(color='rgb(100, 150, 255)'),
        name=course['course_title'][:30] + "..."
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        title=f"Profil du Cours"
    )
    
    # CLÉ UNIQUE AJOUTÉE ICI
    st.plotly_chart(fig, use_container_width=True, key=chart_key)




# =============================================================================
# DASHBOARD STREAMLIT PRINCIPAL
# =============================================================================

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Smart Course Recommender",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .course-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #ff6b6b;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown('<h1 class="main-header">🎓 Smart Course Recommender</h1>', unsafe_allow_html=True)
    st.markdown("### Système de Recommandation Hybride Intelligent pour l'Éducation")
    st.markdown("---")
    
    # Initialisation du système
    if 'recommender' not in st.session_state:
        # Chargement des données
        @st.cache_data
        def load_data():
            try:
                df = pd.read_csv("CourseraDataset-Clean.csv")
                st.success(f"✅ Dataset chargé avec succès! {len(df)} cours disponibles.")
                return df
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du dataset: {e}")
                return pd.DataFrame()
            
        df = load_data()
        if not df.empty:
            st.session_state.recommender = HybridCourseRecommender(df)
            st.session_state.data_loaded = True
        else:
            st.session_state.data_loaded = False
            return
    
    if not st.session_state.get('data_loaded', False):
        st.warning("📁 Le dataset n'a pas pu être chargé. Vérifiez le fichier 'CourseraDataset-Clean.csv'")
        return
    
    # SIDEBAR - CONFIGURATION
    with st.sidebar:
        st.header("⚙️ Configuration du Système")
        
        # Stratégie d'hybridation
        st.subheader("🎯 Stratégie de Recommandation")
        strategy = st.selectbox(
            "Méthode d'hybridation",
            ["cascade", "mixed", "weighted"],
            format_func=lambda x: {
                "cascade": "Cascade (Recommandé)",
                "mixed": "Mixte (Résultats séparés)", 
                "weighted": "Pondérée"
            }[x]
        )
        
        # Section Knowledge-Based
        st.subheader("🎓 Filtres Knowledge-Based")
        level = st.selectbox(
            "Niveau de difficulté",
            ["All", "Beginner", "Intermediate", "Advanced", "Mixed"]
        )
        
        min_rating = st.slider(
            "Note minimale requise",
            min_value=3.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )
        
        max_duration = st.selectbox(
            "Durée maximale",
            ["All", "4 weeks", "8 weeks", "12 weeks", "16 weeks", "20+ weeks"]
        )
        
        # Compétences
        st.subheader("🛠️ Compétences Recherchées")
        skills_input = st.text_input(
            "Compétences (séparées par des virgules)",
            placeholder="Ex: python, machine learning, data analysis"
        )
        required_skills = [s.strip() for s in skills_input.split(',')] if skills_input else []
        
        # Section Collaborative
        st.subheader("❤️ Préférences Personnelles")
        available_courses = st.session_state.recommender.df_clean['course_title'].head(50).tolist()
        liked_courses = st.multiselect(
            "Cours que vous avez appréciés",
            available_courses,
            help="Sélectionnez les cours que vous avez aimés pour des recommandations personnalisées"
        )
        
        # Section Recherche
        st.subheader("🔍 Recherche Sémantique")
        search_query = st.text_input(
            "Description de ce que vous cherchez",
            placeholder="Ex: cours python pour débutants avec projets pratiques"
        )
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Générer Recommendations", type="primary", use_container_width=True):
                st.session_state.generate_recos = True
        with col2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.generate_recos = False
                st.rerun()
    
    # CONTENU PRINCIPAL
    if st.session_state.get('generate_recos', False):
        show_recommendations_page(strategy, level, min_rating, max_duration, required_skills, liked_courses, search_query)
    else:
        show_welcome_page()

def show_welcome_page():
    """Page d'accueil avec présentation du système"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🌟 Bienvenue dans le Smart Course Recommender!")
        
        st.markdown("""
        ### 🎯 Un Système de Recommandation Hybride Avancé
        
        Notre plateforme combine **4 approches intelligentes** pour vous trouver les meilleurs cours :
        """)
        
        # Features
        features = [
            ("🔍 **Recherche Sémantique**", "Comprend le sens de votre recherche, pas juste les mots-clés"),
            ("🎓 **Filtres Intelligents**", "Adapte les recommandations à votre niveau et disponibilité"),
            ("❤️ **Apprentissage Collaboratif**", "S'appuie sur vos préférences pour affiner les suggestions"),
            ("⭐ **Analyse de Popularité**", "Considère les notes et avis de la communauté"),
            ("🤝 **Hybridation Avancée**", "Combine toutes ces approches pour des résultats optimaux")
        ]
        
        for feature, description in features:
            with st.container():
                col_f1, col_f2 = st.columns([1, 4])
                with col_f1:
                    st.markdown(f"**{feature}**")
                with col_f2:
                    st.markdown(description)
            st.write("")
        
        st.info("""
        💡 **Pour commencer :** 
        1. Configurez vos préférences dans la sidebar 
        2. Cliquez sur **'Générer Recommendations'**
        3. Découvrez des cours parfaitement adaptés à vos besoins !
        """)
    
    with col2:
        st.header("📊 Statistiques du Catalogue")
        
        # Métriques globales
        recommender = st.session_state.recommender
        df = recommender.df_clean
        
        if len(df) > 0:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("📚 Cours Disponibles", len(df))
                st.metric("⭐ Note Moyenne", f"{df['rating'].mean():.2f}")
            with col_m2:
                st.metric("🎓 Niveaux", df['level'].nunique())
                st.metric("🏢 Organisations", df['offered_by'].nunique())
            
            # Graphique des niveaux
            level_counts = df['level'].value_counts()
            fig_levels = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                title="Répartition par Niveau"
            )
            st.plotly_chart(fig_levels, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour les statistiques")
    
    # Section témoignages
    st.markdown("---")
    st.header("🎭 Scénarios d'Utilisation")
    
    scenarios = [
        {
            "title": "🚀 Débutant en Programmation",
            "description": "Je veux apprendre Python from scratch avec des projets pratiques",
            "filters": {"level": "Beginner", "skills": ["python", "programming"]}
        },
        {
            "title": "📊 Professionnel en Reconversion", 
            "description": "Je cherche une formation Data Science complète avec certification",
            "filters": {"level": "Intermediate", "duration": "8-12 weeks"}
        },
        {
            "title": "🎯 Spécialisation Avancée",
            "description": "Je veux me perfectionner en Machine Learning avec des cas réels",
            "filters": {"level": "Advanced", "min_rating": 4.5}
        }
    ]
    
    cols = st.columns(3)
    for i, scenario in enumerate(scenarios):
        with cols[i]:
            with st.container():
                st.markdown(f"### {scenario['title']}")
                st.markdown(scenario['description'])
                st.caption("💡 Idéal pour ce type de profil")

def show_recommendations_page(strategy, level, min_rating, max_duration, required_skills, liked_courses, search_query):
    """Page de résultats des recommandations"""
    
    # Préparation des inputs utilisateur
    user_input = {
        'strategy': strategy,
        'search_query': search_query,
        'filters': {
            'level': level if level != "All" else None,
            'min_rating': min_rating,
            'max_duration_weeks': parse_duration(max_duration) if max_duration != "All" else None,
            'required_skills': required_skills
        },
        'liked_courses': liked_courses
    }
    
    # Génération des recommandations
    with st.spinner("🔮 Génération des recommandations personnalisées..."):
        recommendations = st.session_state.recommender.hybrid_recommend(user_input)
    
    # Affichage des résultats
    if strategy == "mixed" and isinstance(recommendations, dict):
        show_mixed_recommendations(recommendations, user_input)
    else:
        show_unified_recommendations(recommendations, user_input)

def show_unified_recommendations(recommendations, user_input):
    """Affiche les recommandations unifiées"""
    
    st.header("📋 Vos Recommandations Personnalisées")
    
    if recommendations.empty:
        st.warning("""
        🤔 Aucune recommandation ne correspond exactement à vos critères.
        
        **Suggestions :**
        - Élargissez vos filtres de recherche
        - Réduisez la note minimale requise  
        - Essayez d'autres mots-clés
        - Consultez les cours tendances ci-dessous
        """)
        
        # Fallback: cours tendances
        st.subheader("🔥 Cours Tendances (Alternative)")
        trending = st.session_state.recommender.knowledge_recommender.get_trending_courses(10)
        display_courses_grid(trending, user_input, show_explanation=False)
        return
    
    # Métriques des résultats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Cours Trouvés", len(recommendations))
    with col2:
        st.metric("⭐ Note Moyenne", f"{recommendations['rating'].mean():.2f}")
    with col3:
        avg_duration = f"{recommendations['duration_weeks'].mean():.1f} semaines"
        st.metric("📅 Durée Moyenne", avg_duration)
    with col4:
        best_rating = recommendations['rating'].max()
        st.metric("🏆 Meilleure Note", f"{best_rating:.1f}")
    
    st.markdown("---")
    
    # Affichage des cours
    display_courses_grid(recommendations, user_input)

def show_mixed_recommendations(recommendations_dict, user_input):
    """Affiche les recommandations par catégorie (stratégie mixte)"""
    
    st.header("🎭 Recommandations par Catégorie")
    
    tabs = st.tabs(["🔍 Sémantique", "🎓 Knowledge-Based", "❤️ Collaboratif", "🔥 Tendance"])
    
    with tabs[0]:
        if 'content_based' in recommendations_dict and not recommendations_dict['content_based'].empty:
            st.subheader("Basé sur votre recherche sémantique")
            display_courses_grid(recommendations_dict['content_based'], user_input)
        else:
            st.info("Aucune recommandation sémantique. Essayez d'ajouter une description de recherche.")
    
    with tabs[1]:
        if 'knowledge_based' in recommendations_dict and not recommendations_dict['knowledge_based'].empty:
            st.subheader("Basé sur vos filtres et contraintes")
            display_courses_grid(recommendations_dict['knowledge_based'], user_input)
        else:
            st.info("Aucun cours ne correspond à tous vos filtres. Essayez de les assouplir.")
    
    with tabs[2]:
        if 'collaborative' in recommendations_dict and not recommendations_dict['collaborative'].empty:
            st.subheader("Basé sur vos cours préférés")
            display_courses_grid(recommendations_dict['collaborative'], user_input)
        else:
            st.info("Sélectionnez des cours que vous avez aimés pour des recommandations collaboratives.")
    
    with tabs[3]:
        if 'trending' in recommendations_dict and not recommendations_dict['trending'].empty:
            st.subheader("Cours populaires en ce moment")
            display_courses_grid(recommendations_dict['trending'], user_input, show_explanation=False)
        else:
            st.info("Chargement des cours tendances...")

# MODIFIEZ la fonction display_courses_grid comme suit :
def display_courses_grid(courses_df, user_input, show_explanation=True):
    """Affiche une grille de cours avec leurs détails"""
    
    if courses_df.empty:
        st.write("Aucun cours à afficher.")
        return
    
    max_values = {
        'max_reviews': courses_df['number_of_review'].max(),
        'max_duration': courses_df['duration_weeks'].max()
    }
    
    for idx, course in courses_df.iterrows():
        with st.container():
            # Header du cours
            col_header1, col_header2 = st.columns([3, 1])
            
            with col_header1:
                st.subheader(f"📚 {course['course_title']}")
            
            with col_header2:
                # Badges
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.metric("⭐", f"{course['rating']:.1f}")
                with col_b2:
                    st.metric("👥", f"{course['number_of_review']}")
                with col_b3:
                    st.metric("📅", f"{int(course['duration_weeks'])}s")
            
            # Informations détaillées
            col_info1, col_info2 = st.columns([2, 1])
            
            with col_info1:
                st.write(f"**🏢 Organisme :** {course['offered_by']}")
                st.write(f"**👨‍🏫 Instructeur :** {course['instructor']}")
                st.write(f"**🎯 Niveau :** {course['level']}")
                st.write(f"**🛠️ Compétences :** {course['skill_gain']}")
                
                if 'what_you_will_learn' in course and pd.notna(course['what_you_will_learn']) and course['what_you_will_learn'] != '':
                    with st.expander("📖 Ce que vous apprendrez"):
                        st.write(course['what_you_will_learn'])
            
            with col_info2:
                # UTILISATION DE LA NOUVELLE FONCTION AVEC CLÉ UNIQUE
                chart_key = f"radar_{course['course_title'][:20]}_{idx}"
                display_radar_chart(course, max_values, chart_key)
            
            # Lien vers le cours
            if 'course_url' in course and pd.notna(course['course_url']):
                st.markdown(f"[🔗 Accéder au cours sur Coursera]({course['course_url']})")
            
            # Explication de la recommandation
            if show_explanation:
                with st.expander("💡 Pourquoi ce cours vous est recommandé", expanded=False):
                    explanation = generate_explanation(course, user_input)
                    st.write(explanation)
            
            st.markdown("---")

def parse_duration(duration_str):
    """Convertit la durée en semaines"""
    if duration_str == "All":
        return None
    elif duration_str == "20+ weeks":
        return 20
    else:
        return int(duration_str.split()[0])

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()