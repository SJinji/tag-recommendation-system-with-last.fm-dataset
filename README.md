# Tag Recommendation System

This project is focused on creating a recommendation system for tags related to music artists. It employs different techniques, including Word2Vec and BERT, to train models that generate tag recommendations. The code covers preprocessing of tags, training Word2Vec models, hyperparameter tuning, visualization of tag similarity, and an experimentation using BERT.

## Key Components

### 1. **Preprocessing Tags**
   - Cleaning and formatting tags, including:
     - Converting to lowercase.
     - Tokenizing.
     - Removing stopwords.
     - Removing special characters.
     - Removing single-character tags.
     - Removing the tag 'none'.
       
### 2. **Visualization**
   - Word Clouds to represent common themes across the dataset.

### 3. **Word2Vec Model Training**
   - Training a Word2Vec model using the Gensim library.
   - Normalizing the tag embeddings.
   - Hyperparameter tuning to find the best configuration for vector size, window, and minimum count.

### 4. **BERT Embeddings**
   - Utilizing the pre-trained BERT model to create embeddings for the tags.
   - Tokenizing and encoding tags using BERT's tokenizer.
   - Creating PyTorch DataLoader for handling input tags and attention masks.
   - Normalizing the BERT tag embeddings.

### 5. **Evaluation and Visualization Techniques**
To assess the performance and understand the effectiveness of the tag recommendation system, the following evaluation and visualization techniques are used:

-  Evaluation Metrics:
    - **Precision, Recall, and F1-score**: These metrics provide a quantitative evaluation of the recommendation system's accuracy.

- Visualization Techniques:
    - **Random Tag Similarity Heatmap**: A heatmap showing similarities between randomly selected tags, helping visualize tag relationships. 
    - **User-Specific Tag Similarity Heatmap**: Displays similarities between an artist's true tags and the recommended tags, assisting in understanding how recommendations align with user preferences. 
    - **Ranked Positions of True Tags Histogram**: A histogram showing the ranked positions of true tags among recommendations, giving insights into the quality of recommendations. 

The combination of traditional evaluation metrics and visualization techniques provides both a quantitative and visual understanding of the recommendation system's performance, enabling a more comprehensive analysis.

## Libraries and Tools Used
   - Pandas and NumPy for data manipulation.
   - NLTK for text preprocessing and tokenization.
   - Gensim for Word2Vec modeling.
   - Scikit-learn for similarity calculations and metrics.
   - Transformers (Hugging Face) for BERT tokenizer and model.
   - PyTorch for GPU handling and data loading.
   - Seaborn and Matplotlib for visualization.

## Usage
To use the code, ensure that you have the required libraries installed and follow the functions and scripts provided. Preprocess the tag data, train the Word2Vec model or use BERT for embeddings, and apply visualization or evaluation methods as needed.

## References
[1] [https://towardsdatascience.com/making-music-tagging-ai-explainable-through-source-separation-2d9493547a7e](https://www.intechopen.com/chapters/64576)https://www.intechopen.com/chapters/64576  
[2] https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05  
[3] https://github.com/yaxue1123/data-analysis-of-last.fm  
[4] https://github.com/Crone1/Last.fm-Recommender-System/tree/main  
[5] https://github.com/cipher813/recommender_system/tree/master  
[6] https://github.com/JohnsonKuan/movie-rec-tags/blob/master/Movie-Rec-Movielens-Tags.ipynb  
[7] https://www.kaggle.com/code/sobrinomario/a-tag-based-recommender-system-for-careervillage  
