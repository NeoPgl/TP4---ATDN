#Exercice 1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC  # SVC pour probabilités (Exercice 2)
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score





print("======================================Exercice 1 : ===================================================\n")




# 1. Charger le jeu de données
url = "spam.csv"  


# Nettoyage et Rename des colonnes en label et message
data = pd.read_csv(url, encoding='latin-1', header=None, names=['label', 'message'], usecols=[0, 1])


# Prétraitement des données

# Convertir le texte en minuscules et enlever les caractères non-alphabétiques
data['message'] = data['message'].str.replace(r'\W+', ' ', regex=True).str.lower()

# Convertir les étiquettes "ham" et "spam" en 0 et 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Supprimer les valeurs NaN existant dans la colonne 'label'
if data['label'].isnull().any():
    data = data.dropna(subset=['label'])

# Vérification après nettoyage
print(f"Après nettoyage, nombre de messages : {len(data)}")
print(data.head())  # Afficher les 5 premières lignes pour vérifier

# Vectorisation des messages
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']


#2. Division des données en ensemble d'entraînement et de test en 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #Utilisation de test_size=0.3, définit 30% des données en test et le reste en entrainement


# 3. Entraînement du modèle SVM avec un noyau linéaire
start_time = time.time()  # Temps de départ pour le SVM
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train, y_train)


# Prédictions et évaluation du modèle
y_pred_svm = svm_model.predict(X_test)

svm_time = time.time() - start_time  # Temps écoulé pour le SVM


# Affichage des performances du modèle
print("\n================================\n")
print("SVM Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))


# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot()
plt.show()



# Calcul des scores de décision pour les prédictions
y_scores = svm_model.decision_function(X_test)

# Génération des données pour la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

#5. Tracé de la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Courbe ROC (AUC = {roc_auc:.2f})", color='red')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.5)")
plt.title("Courbe ROC-AUC")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Affichage du score AUC
print(f"AUC-ROC Score: {roc_auc:.2f}")




#Exercice 2 : 


print("======================================Exercice 2 : ===================================================\n")


# Entraînement du modèle Naïve Bayes
start_time = time.time()  # Temps de départ pour Naïve Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
nb_time = time.time() - start_time  # Temps écoulé pour Naïve Bayes
print(f"Temps d'exécution du modèle Naïve Bayes : {nb_time:.4f} secondes")



# Entraînement du modèle de régression logistique
start_time = time.time()  # Temps de départ pour Régression Logistique
log_reg_model = LogisticRegression(max_iter=1000) 
log_reg_model.fit(X_train, y_train)
y_pred_lr = log_reg_model.predict(X_test)
lr_time = time.time() - start_time  # Temps écoulé pour Régression Logistique
print(f"Temps d'exécution du modèle Régression Logistique : {lr_time:.4f} secondes")



# Comparaison des performances des modèles
performance_data = {
    'Model': ['Naive Bayes', 'Régression Logistique', 'SVM'],
    'Precision': [
        classification_report(y_test, y_pred, output_dict=True)['macro avg']['precision'],
        classification_report(y_test, y_pred_lr, output_dict=True)['macro avg']['precision'],
        classification_report(y_test, y_pred_svm, output_dict=True)['macro avg']['precision'],
    ],
    'Rappel': [
        classification_report(y_test, y_pred, output_dict=True)['macro avg']['recall'],
        classification_report(y_test, y_pred_lr, output_dict=True)['macro avg']['recall'],
        classification_report(y_test, y_pred_svm, output_dict=True)['macro avg']['recall'],
    ],
    'F1-Score': [
        classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score'],
        classification_report(y_test, y_pred_lr, output_dict=True)['macro avg']['f1-score'],
        classification_report(y_test, y_pred_svm, output_dict=True)['macro avg']['f1-score'],
    ]
}

# Convertir en DataFrame pour une meilleure lisibilité
performance_df = pd.DataFrame(performance_data)
print(performance_df)



print("\n====================================== Voting Classifier vote 'hard': ===================================================\n")

# 1. Initialisation des modèles de base pour le Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('nb', nb_model),  # Naive Bayes
        ('lr', log_reg_model),  # Régression Logistique
        ('svm', svm_model)  # SVM
    ],
    voting='hard'
)

# Entraînement du Voting Classifier
voting_clf.fit(X_train, y_train)

# Prédictions avec le Voting Classifier
y_pred_voting = voting_clf.predict(X_test)

# Évaluation des performances du Voting Classifier
print("Voting Classifier Performance (vote hard):")
print("Précision:", accuracy_score(y_test, y_pred_voting))
print("Classification Report:\n", classification_report(y_test, y_pred_voting))


print("\n====================================== Voting Classifier vote 'soft': ===================================================\n")

# SVM n'a pas de méthode pour prédire les probabilités par défaut,on remplace SVM par une version avec probabilités (import SVC plus haut).

svm_model_prob = SVC(kernel='linear', probability=True, random_state=42)
svm_model_prob.fit(X_train, y_train)

voting_clf_soft = VotingClassifier(
    estimators=[
        ('nb', nb_model),  # Naive Bayes
        ('lr', log_reg_model),  # Régression Logistique
        ('svm', svm_model_prob)  # SVM avec probabilités
    ],
    voting='soft'
)

# Entraînement du Voting Classifier avec vote soft
voting_clf_soft.fit(X_train, y_train)

# Prédictions avec le Voting Classifier (soft)
y_pred_voting_soft = voting_clf_soft.predict(X_test)

# Évaluation des performances du Voting Classifier (soft)
print("Voting Classifier Performance (vote soft):")
print("Précision:", accuracy_score(y_test, y_pred_voting_soft))
print("Classification Report:\n", classification_report(y_test, y_pred_voting_soft))


#====================================== Courbes ROC-AUC pour les modèles : ===================================================



# Calculer les probabilités ou scores pour chaque modèle individuel
y_scores_nb = nb_model.predict_proba(X_test)[:, 1]
y_scores_lr = log_reg_model.predict_proba(X_test)[:, 1]
y_scores_svm = svm_model_prob.decision_function(X_test)
y_scores_voting_soft = voting_clf_soft.predict_proba(X_test)[:, 1]

# Génération des courbes ROC
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_scores_nb)
roc_auc_nb = roc_auc_score(y_test, y_scores_nb)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
roc_auc_lr = roc_auc_score(y_test, y_scores_lr)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_scores_svm)
roc_auc_svm = roc_auc_score(y_test, y_scores_svm)

fpr_voting, tpr_voting, _ = roc_curve(y_test, y_scores_voting_soft)
roc_auc_voting = roc_auc_score(y_test, y_scores_voting_soft)

# Tracer les courbes ROC
plt.figure(figsize=(10, 8))

# Courbes pour chaque modèle
plt.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {roc_auc_nb:.2f})", color='blue')
plt.plot(fpr_lr, tpr_lr, label=f"Régression Logistique (AUC = {roc_auc_lr:.2f})", color='green')
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.2f})", color='orange')
plt.plot(fpr_voting, tpr_voting, label=f"Voting Classifier (Soft) (AUC = {roc_auc_voting:.2f})", color='red')

# Ligne de base
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.5)")

# Paramètres du graphique
plt.title("Courbes ROC-AUC pour les modèles individuels")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()





# #Exercice 3 : 


# print("======================================Exercice 3 : ===================================================\n")


# # 2. Entraînement du modèle GMM
# gmm_model = GaussianMixture(n_components=2, random_state=42)  # 2 classes : ham (0) et spam (1)
# gmm_model.fit(X_train.toarray())  # Entraînement du modèle avec les données vectorisées (on doit convertir X_train en array)

# # Prédictions sur l'ensemble de test
# y_pred_gmm = gmm_model.predict(X_test.toarray())  # Prédictions sur l'ensemble de test

# # 3. Évaluation des performances du modèle
# print("Gaussian Mixture Model Performance:")
# print("Accuracy:", accuracy_score(y_test, y_pred_gmm))
# print("Classification Report:\n", classification_report(y_test, y_pred_gmm))



# # 5. Tracer les frontières de décision du GMM

# # Réduction des dimensions des données avec PCA à 2 dimensions pour la visualisation
# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train.toarray())  # Appliquer PCA aux données d'entraînement


# # Créer une grille de points pour les prédictions
# x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
# y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))

# # Appliquer le GMM à la grille de points
# grid_points = np.c_[xx.ravel(), yy.ravel()]
# Z_gmm = gmm_model.predict(pca.inverse_transform(grid_points))
# Z_gmm = Z_gmm.reshape(xx.shape)

# # Tracer les frontières de décision pour le GMM
# plt.contourf(xx, yy, Z_gmm, alpha=0.8, cmap=plt.cm.coolwarm)

# # Tracer les points d'entraînement
# plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
# plt.title("Frontières de décision du GMM (Réduit avec PCA)")
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()




# #Exercice 4 : 

# print("======================================Exercice 4 : ===================================================\n")


# # Définir une grille de paramètres pour C et kernel
# param_grid = {
#     'C': [0.1, 1, 10, 100],  # Tester différentes valeurs de C
#     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Tester différents types de noyaux
# }

# # Créer une instance du modèle SVM
# svm = SVC(random_state=42)

# # Créer l'objet GridSearchCV pour rechercher les meilleurs paramètres
# grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # Entraîner GridSearchCV
# grid_search.fit(X_train, y_train)

# # Afficher les meilleurs paramètres et la meilleure performance
# print(f"\nBest parameters found: {grid_search.best_params_}")
# print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

# # Utiliser le meilleur modèle trouvé pour faire des prédictions sur l'ensemble de test
# best_svm_model = grid_search.best_estimator_
# y_pred_best_svm = best_svm_model.predict(X_test)

# # Évaluation du modèle avec les meilleurs paramètres
# print("\nSVM avec GridSearchCV Performance (avec meilleurs paramètres):")
# print("Accuracy:", accuracy_score(y_test, y_pred_best_svm))
# print("Classification Report:\n", classification_report(y_test, y_pred_best_svm))

# # Matrice de confusion
# cm_best = confusion_matrix(y_test, y_pred_best_svm)
# disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=['Ham', 'Spam'])
# disp_best.plot()
# plt.show()

# # Calcul des scores de décision pour les prédictions
# y_scores_best_svm = best_svm_model.decision_function(X_test)

# # Génération des données pour la courbe ROC
# fpr_best_svm, tpr_best_svm, thresholds_best_svm = roc_curve(y_test, y_scores_best_svm)
# roc_auc_best_svm = roc_auc_score(y_test, y_scores_best_svm)

# # Tracé de la courbe ROC
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_best_svm, tpr_best_svm, label=f"Courbe ROC (AUC = {roc_auc_best_svm:.2f})", color='blue')
# plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.5)")
# plt.title("Courbe ROC-AUC - SVM avec GridSearchCV")
# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)
# plt.show()

# # Affichage du score AUC
# print(f"AUC-ROC Score pour le modèle optimisé: {roc_auc_best_svm:.2f}")



print("======================================Exercice 5 : ===================================================\n")



#Exercice 5 : 


# 7. Calcul des métriques de performance
precision_nb = precision_score(y_test, y_pred)
recall_nb = recall_score(y_test, y_pred)
f1_nb = f1_score(y_test, y_pred)
roc_auc_nb = roc_auc_score(y_test, y_scores_nb)

precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_scores_lr)

precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_scores_svm)

precision_voting = precision_score(y_test, y_pred_voting_soft)
recall_voting = recall_score(y_test, y_pred_voting_soft)
f1_voting = f1_score(y_test, y_pred_voting_soft)
roc_auc_voting = roc_auc_score(y_test, y_scores_voting_soft)


# Tableau comparatif des performances
performance_comparison = {
    'Model': ['Naive Bayes', 'Logistic Regression', 'SVM', 'Voting Classifier (Soft)'],
    'Precision': [precision_nb, precision_lr, precision_svm, precision_voting],
    'Recall': [recall_nb, recall_lr, recall_svm, recall_voting],
    'F1-Score': [f1_nb, f1_lr, f1_svm, f1_voting],
    'AUC-ROC': [roc_auc_nb, roc_auc_lr, roc_auc_svm, roc_auc_voting]
}

# Convertir en DataFrame pour une meilleure lisibilité
performance_df = pd.DataFrame(performance_comparison)
print(performance_df)




# Temps d'exécution total des modèles
total_time = nb_time + lr_time + svm_time
print(f"Temps total d'exécution des modèles : {total_time:.4f} secondes")