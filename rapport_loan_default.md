# Rapport d'Analyse — Prédiction du Défaut de Remboursement de Prêt Automobile

> **Source des données :** Kaggle — *Vehicle Loan Default Prediction* (avikpaul4u)
> **Type de problème :** Classification binaire supervisée
> **Variable cible :** `LOAN_DEFAULT` (0 = pas de défaut, 1 = défaut)
# FOUAD NOUR AL HOUDA 
![photo de nour al houda  (1)](https://github.com/user-attachments/assets/10d95830-7bde-4515-aebb-a451213a2293)
---

## 1. Contexte et Objectif

Ce projet vise à prédire si un emprunteur va faire défaut sur un prêt automobile. La prédiction précoce du défaut de remboursement est cruciale pour les institutions financières afin de limiter les pertes et d'optimiser les décisions d'octroi de crédit.

---

## 2. Description des Données

### 2.1 Dimensions du jeu de données

| Dataset      | Lignes  | Colonnes |
|-------------|---------|----------|
| Entraînement | 233 154 | 41       |
| Test         | 112 392 | 40       |

### 2.2 Variables disponibles

Le jeu de données contient 41 colonnes couvrant plusieurs dimensions du profil emprunteur :

| Catégorie | Variables |
|-----------|-----------|
| Identifiant | `UNIQUEID` |
| Informations prêt | `DISBURSED_AMOUNT`, `ASSET_COST`, `LTV`, `DISBURSAL_DATE` |
| Localisation | `BRANCH_ID`, `SUPPLIER_ID`, `MANUFACTURER_ID`, `CURRENT_PINCODE_ID`, `STATE_ID` |
| Profil emprunteur | `DATE_OF_BIRTH`, `EMPLOYMENT_TYPE`, `EMPLOYEE_CODE_ID` |
| Documents identité | `MOBILENO_AVL_FLAG`, `AADHAR_FLAG`, `PAN_FLAG`, `VOTERID_FLAG`, `DRIVING_FLAG`, `PASSPORT_FLAG` |
| Score de crédit | `PERFORM_CNS_SCORE`, `PERFORM_CNS_SCORE_DESCRIPTION` |
| Comptes primaires | `PRI_NO_OF_ACCTS`, `PRI_ACTIVE_ACCTS`, `PRI_OVERDUE_ACCTS`, `PRI_CURRENT_BALANCE`, `PRI_SANCTIONED_AMOUNT`, `PRI_DISBURSED_AMOUNT` |
| Comptes secondaires | `SEC_NO_OF_ACCTS`, `SEC_ACTIVE_ACCTS`, `SEC_OVERDUE_ACCTS`, `SEC_CURRENT_BALANCE`, `SEC_SANCTIONED_AMOUNT`, `SEC_DISBURSED_AMOUNT` |
| Historique crédit | `AVERAGE_ACCT_AGE`, `CREDIT_HISTORY_LENGTH`, `NO_OF_INQUIRIES`, `NEW_ACCTS_IN_LAST_SIX_MONTHS`, `DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS` |
| Mensualités | `PRIMARY_INSTAL_AMT`, `SEC_INSTAL_AMT` |
| **Cible** | **`LOAN_DEFAULT`** |

---

## 3. Analyse Exploratoire des Données (EDA)

### 3.1 Distribution de la variable cible

Le jeu d'entraînement présente un **déséquilibre de classes** notable :

| Classe | Effectif | Proportion |
|--------|----------|------------|
| Pas de défaut (0) | 182 543 | **78.29 %** |
| Défaut (1)        | 50 611  | **21.71 %** |

Ce déséquilibre est important à considérer lors de la modélisation et de l'évaluation des performances.

### 3.2 Analyse par type d'emploi

| Type d'emploi | Taux de non-défaut | Taux de défaut |
|---------------|-------------------|----------------|
| Salaried      | 79.65 %           | 20.35 %        |
| Self employed | 77.23 %           | **22.77 %**    |
| Missing       | 78.54 %           | 21.46 %        |

Les travailleurs indépendants (*Self employed*) présentent un taux de défaut légèrement plus élevé.

### 3.3 Analyse par documents d'identité

| Document | Modalité | Taux de défaut |
|----------|----------|----------------|
| Aadhar Flag | Absent (0) | **25.64 %** |
| Aadhar Flag | Présent (1) | 20.96 % |
| Voter ID Flag | Présent (1) | **26.09 %** |
| Voter ID Flag | Absent (0) | 20.96 % |
| Passport | Présent (1) | **14.92 %** ✅ |
| Passport | Absent (0) | 21.72 % |
| Driving Flag | Présent (1) | 20.15 % ✅ |
| Driving Flag | Absent (0) | 21.74 % |

La possession d'un **passeport** est associée à un risque de défaut significativement plus faible, ce qui peut indiquer un profil socio-économique plus stable.

### 3.4 Analyse du montant décaissé (`DISBURSED_AMOUNT`)

| Statistique | Valeur |
|-------------|--------|
| Nombre d'observations | 233 154 |
| Moyenne | 54 356.99 |
| Écart-type | 12 971.29 |
| Nombre de valeurs aberrantes (±3σ) | **3 076** |

Les valeurs aberrantes ont été imputées par la moyenne pour éviter leur impact sur les modèles.

### 3.5 Discrétisation du montant décaissé

La variable `DISBURSED_AMOUNT` a été transformée en variable catégorielle par quartiles :

| Catégorie | Effectif |
|-----------|----------|
| Medium    | 58 676   |
| Low       | 58 537   |
| Extreme   | 58 207   |
| High      | 57 734   |

---

## 4. Préparation et Ingénierie des Features

### 4.1 Nettoyage des données
- Suppression des doublons dans les ensembles train et test.
- Imputation des valeurs aberrantes de `DISBURSED_AMOUNT` par la moyenne (méthode 3σ).

### 4.2 Feature Engineering
- **Binning** de `DISBURSED_AMOUNT` en 4 catégories ordinales : Low, Medium, High, Extreme.
- **Encodage One-Hot** des variables catégorielles : `EMPLOYMENT_TYPE`, `PERFORM_CNS_SCORE_DESCRIPTION`, `DISBURSED_AMOUNT_bins`.

### 4.3 Variables supprimées
Les colonnes suivantes ont été retirées du jeu de features :

| Variable retirée | Raison |
|-----------------|--------|
| `LOAN_DEFAULT` | Variable cible |
| `UNIQUEID` | Identifiant non informatif |
| `DISBURSED_AMOUNT` | Remplacée par `DISBURSED_AMOUNT_bins` |
| `ASSET_COST` | Colinéarité potentielle |
| `DISBURSAL_DATE` | Variable temporelle brute |

### 4.4 Découpage train/test

| Paramètre | Valeur |
|-----------|--------|
| Taille de l'ensemble de test | 30 % |
| `random_state` | 42 |
| Stratification | Oui (`stratify=y`) |

---

## 5. Modélisation

### 5.1 Modèles utilisés

Deux algorithmes de classification ont été appliqués :

**Régression Logistique**
- Solver : `liblinear`
- Évaluation : Validation croisée stratifiée (5 folds)

**Random Forest Classifier**
- Nombre d'arbres : 10 estimateurs
- Évaluation : Prédiction directe sur l'ensemble de test

### 5.2 Validation croisée

| Paramètre | Valeur |
|-----------|--------|
| Méthode | `StratifiedKFold` |
| Nombre de folds | 5 |
| Shuffle | Oui |
| Scoring | Accuracy |

---

## 6. Résultats et Performances

### 6.1 Matrice de confusion — Random Forest

|                  | Prédit : Non-défaut | Prédit : Défaut |
|------------------|--------------------:|----------------:|
| **Réel : Non-défaut** | 51 423 | 3 341 |
| **Réel : Défaut**     | 13 753 | 1 430 |

### 6.2 Comparaison des modèles

| Modèle | Méthode d'évaluation | Accuracy |
|--------|---------------------|----------|
| Random Forest | Test set | **76 %** |
| Régression Logistique | Cross-validation (5-fold) | **78 %** |

La Régression Logistique obtient une accuracy légèrement supérieure en validation croisée, suggérant une meilleure généralisation que le Random Forest (limité à 10 arbres dans cette configuration).

### 6.3 Observations sur les performances

- Le taux de **faux négatifs** (13 753 défauts non détectés) est élevé, ce qui est problématique dans un contexte de risque financier où manquer un défaut coûte cher.
- L'accuracy seule ne suffit pas dans un contexte déséquilibré : des métriques complémentaires comme le **Rappel**, la **Précision** et l'**AUC-ROC** seraient plus pertinentes.

---

## 7. Bibliothèques Utilisées

| Bibliothèque | Usage |
|-------------|-------|
| `pandas`, `numpy` | Manipulation des données |
| `matplotlib`, `seaborn` | Visualisations statiques |
| `plotly` | Visualisations interactives |
| `scikit-learn` | Modélisation, évaluation, encodage |
| `xgboost` | Importé (boosting gradient) |
| `scipy` | Statistiques descriptives |

---

## 8. Conclusion et Perspectives

### Résumé

Ce projet met en place un pipeline complet de Machine Learning pour la prédiction de défaut de prêt automobile : chargement des données, analyse exploratoire, nettoyage, ingénierie des features, modélisation et évaluation. Les deux modèles testés atteignent une accuracy autour de **76–78 %**.

### Pistes d'amélioration

1. **Gérer le déséquilibre des classes** via SMOTE, class_weight ou sous-échantillonnage.
2. **Optimiser les hyperparamètres** du Random Forest (augmenter `n_estimators`, régler `max_depth`).
3. **Utiliser XGBoost** (déjà importé) qui est souvent plus performant sur des données tabulaires déséquilibrées.
4. **Évaluer avec des métriques adaptées** : AUC-ROC, F1-score, rappel, courbe Precision-Recall.
5. **Analyser l'importance des features** pour identifier les variables les plus prédictives du défaut.
6. **Traiter la date de naissance** pour en extraire l'âge du demandeur, potentiellement prédictif.

---

*Rapport généré à partir du notebook : `Predicting_Loan_Default__Classification_.ipynb`*
