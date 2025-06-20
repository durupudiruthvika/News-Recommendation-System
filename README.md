 
# **News Recommendation System**

## **Project Overview**

The **News Recommendation System** aims to provide personalized news article recommendations to users based on their preferences and historical interactions. This system utilizes machine learning techniques to analyze and suggest relevant news articles, improving user engagement and satisfaction.

---

## **Features**

- **🔍 Content-Based Recommendation**: 
  - Articles are recommended based on their similarity (e.g., title, content, keywords).

- **🤖 Filtering Approaches**: 
  - Multiple filtering approaches are explored in the notebook:
    - **Content-Based Filtering**: Recommends articles similar to a given article using techniques like **TF-IDF** and **cosine similarity**.
    - **Collaborative Filtering**: Suggests articles based on user interaction or behavioral data.
    - **Hybrid Approach**: Combines content-based and collaborative filtering for improved accuracy.

- **📊 Data Preprocessing**:
  - Handles cleaning, tokenization, and vectorization of the news data.

- **📈 Evaluation**:
  - Measures the system's performance using metrics such as **accuracy**, **precision**, or **recall**.

- **⚡ Interactive Visualizations**:
  - Includes visualizations to analyze results and model performance.

---

## **Dataset**

The recommendation system relies on a structured dataset containing:

- **Article ID**: Unique identifier for each news article.
- **Title**: Headline of the news article.
- **Content**: Full text or a summary of the news article.
- **Category**: Tag or label associated with the article (e.g., sports, politics, technology).
- **User Interactions**: Optional; historical user data like clicks or ratings.

---

## **Technologies Used**

The following tools and libraries are used in this project:

| Tool/Library         | Purpose                                |
|----------------------|----------------------------------------|
| **Python**           | Programming language                   |
| **Jupyter Notebook** | Code development environment           |
| **Pandas**           | Data manipulation and analysis         |
| **Scikit-learn**     | Machine learning implementation        |
| **NLTK**             | Natural Language Processing (NLP)      |
| **Surprise**         | Collaborative filtering library        |

---

## **Library Imports**

Libraries such as **NumPy**, **Pandas**, and **os** are imported for numerical computations, data manipulation, and file handling.

---

## **Data Loading**

Two datasets are loaded from the `MINDsmall_train` folder:

- **behaviors.tsv**: Contains user interaction data, including user IDs, browsing history, and impressions.
- **news.tsv**: Contains news article information like titles, abstracts, categories, and entity data.

---

### **The Behaviors Dataset**:

- Columns like **Impression ID**, **User ID**, **History**, and **Impressions** are assigned.
- The structure and content of the data are explored.

### **The News Dataset**:

- Columns such as **Title**, **Abstract**, and **Category** are extracted.

---

## **Exploratory Data Analysis (EDA)**

The datasets are analyzed to:

- Understand the data structure.
- Identify key metrics such as user browsing behavior and article distribution across categories.

---

## **Building the Recommendation System**

### **Filtering Approaches:**

- **Content-Based Filtering**:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to vectorize the Title or Abstract text.
  - **Cosine similarity** is computed between articles to recommend similar news.

### **Evaluation Metrics:**

- **CTR (Click-Through Rate)**: Measures click behavior.
- **Diversity**: Normalized count of unique categories per user.
- **Coverage**: Average recommendation coverage over all users.
- **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first clicked item.

---

### **Collaborative Filtering**:

1. **User-Based**: Suggests articles based on user similarity.
2. **Item-Based**: Suggests articles similar to those previously interacted with.

A user-item interaction matrix is created based on user behavior history and impressions. This filtering implementation uses algorithms like **KNNBasic**, **SVD**, and **SVDpp** from the **Surprise** library to build and evaluate the recommender system with **hyperparameter tuning** and **cross-validation**.

#### **Evaluation Metric**:

- **RMSE (Root Mean Squared Error)**: Measures the accuracy of predicted ratings.

---

### **Hybrid Recommendation System**:

- **Collaborative Filtering**:
  - **Singular Value Decomposition (SVD)** is applied to predict user-item ratings from the interaction matrix.

- **Content-Based Filtering**:
  - **TF-IDF** vectorization is used to compute cosine similarity between article titles and abstracts.

- **Hybrid Recommendations**:
  - Combines collaborative filtering and content similarity scores with weighted contributions to generate final recommendations.

---

### **Translation Support**:

- Recommendations can be translated into Telugu, Hindi, Malayalam, or Bengali using the **translatepy** library.

---

## **Evaluation Metrics for Hybrid Model**:

- **RMSE (Root Mean Squared Error)**: Measures the accuracy of predicted ratings.
- **Hit Ratio**: Determines the proportion of relevant recommendations in the top-N list.
- **NDCG (Normalized Discounted Cumulative Gain)**: Evaluates ranking quality using discounted cumulative gain.

---

## Team Members 
Naga Ruthvika Durupudi

Adrija Adhikary

Nagasarapu Sarayu Krishna

Vedha Pranava Mateti

### **Mentor**

- Nandu C Nair
---
 
