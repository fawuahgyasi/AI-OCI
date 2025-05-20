import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import re
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from collections import Counter
import tiktoken
import argparse
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import colorsys
import matplotlib.colors as mcolors
import itertools
import os 
from sklearn.exceptions import NotFittedError

# Set the paths for the files and tokens
results_path = 'results_2'
task_statements_file = 'Task_statements.xlsx'
llm_capabilities_file = 'DATA/ai_capabilities_0830.txt'
wage_file = 'DATA/wage_2023.xlsx'
major_occupations_file = 'DATA/major_occupations.xlsx'
aioe_file = 'DATA/AIOE_csv.csv'
huggingface_token = 'Add your token'
api_key = 'add your key'

# Initialize tokenizer and model for E5
e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct', token=huggingface_token)
e5_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct', token=huggingface_token)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Load the BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def generate_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to compute E5 embeddings
def compute_e5_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings.flatten()

# Function to compute embeddings using the OpenAI API with retries
@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def compute_openai_embeddings(text):
    truncated_text = text[:8192]
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=truncated_text
    )
    embedding = np.array(response.data[0].embedding)
    return embedding

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def compute_openai_embeddings_lg(text):
    truncated_text = text[:8192]
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=truncated_text
    )
    embedding = np.array(response.data[0].embedding)
    return embedding

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def compute_openai_embeddings_sm(text):
    truncated_text = text[:8192]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=truncated_text
    )
    embedding = np.array(response.data[0].embedding)
    return embedding


# Load and preprocess capabilities, assigning unique IDs
def load_and_preprocess_capabilities(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        llm_capabilities = [{'text': line.strip(), 'ai_unique_id': f"cap_{i}"} for i, line in enumerate(file) if line.strip()]
    return llm_capabilities

def compute_cosine_similarity(vec_a, vec_b):
    if not isinstance(vec_a, np.ndarray):
        vec_a = np.array(vec_a)
    if not isinstance(vec_b, np.ndarray):
        vec_b = np.array(vec_b)
    return 1 - cosine(vec_a.flatten(), vec_b.flatten())



# Update to handle unique IDs in embedding generation
def generate_all_embeddings(llm_capabilities):
    embeddings_data = []
    for cap in llm_capabilities:
        text = preprocess_text(cap['text'])
        ai_unique_id = cap['ai_unique_id']
        
        # Generate embeddings for each model
        bert_embedding = generate_embeddings(text, bert_tokenizer, bert_model)
        openai_embedding = compute_openai_embeddings(text)
        openai_embedding_lg = compute_openai_embeddings_lg(text)
        openai_embedding_sm = compute_openai_embeddings_sm(text)
        e5_embedding = compute_e5_embeddings(text, e5_tokenizer, e5_model)
        
        embeddings_data.append({
            'ai_unique_id': ai_unique_id,
            'bert': bert_embedding,
            'openai': openai_embedding,
            'openai_lg': openai_embedding_lg,
            'openai_sm': openai_embedding_sm,
            'e5': e5_embedding
        })
    
    return embeddings_data


def normalize_column(df, column_name):
    df[f'Normalized_{column_name}'] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min()) * 100
    return df

def combine_pca_files(pca_size, pca_percent, embeddings_array, embeddings_array_openai, embeddings_array_openai_lg, embeddings_array_openai_sm, embeddings_array_e5):
    combined_df = pd.DataFrame()

    for prefix, embeddings, compute_embeddings_func, tokenizer, model in zip(
            ['bert', 'openai', 'openai_lg', 'openai_sm', 'e5'], 
            [embeddings_array, embeddings_array_openai, embeddings_array_openai_lg, embeddings_array_openai_sm, embeddings_array_e5],
            [generate_embeddings, compute_openai_embeddings, compute_openai_embeddings_lg, compute_openai_embeddings_sm, compute_e5_embeddings],
            [bert_tokenizer, None, None, None, e5_tokenizer],
            [bert_model, None, None, None, e5_model]
        ):
        embeddings_pca, pca_model = perform_pca(embeddings, pca_size, pca_percent)
        optimal_clusters, centroids = cluster_embeddings(embeddings_pca, len(embeddings_pca[0]), prefix)
        task_statements_df = generate_task_embeddings(task_statements_file, embeddings_pca, centroids, len(embeddings_pca[0]), prefix, pca_model, compute_embeddings_func, tokenizer, model)
        combined_df = pd.concat([combined_df, task_statements_df], axis=1)

    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    # combined_df.to_csv(f'{results_path}task_embeddings_cosine_normal_similarity_revised_pca_{pca_size if pca_size > 0 else pca_percent}.csv', index=False)
    combined_df.to_csv(os.path.join(results_path, f'task_embeddings_cosine_normal_similarity_revised_pca_{pca_size if pca_size > 0 else pca_percent}.csv'), index=False)

def perform_pca(embeddings, pca_size, pca_percent=0.0, target_variance=0.95):
    """
    Perform PCA on the provided embeddings.
    """
    max_components = min(embeddings.shape[0], embeddings.shape[1]) - 1  # -1 to exclude the first component

    if pca_size > 0:
        num_components = min(pca_size + 1, max_components)  # Ensure it's within the allowable range
    else:
        pca = PCA(n_components=None)
        pca.fit(embeddings)
        explained_variance_ratio = pca.explained_variance_ratio_[1:]  # Exclude the first component
        cumulative_variance = np.cumsum(explained_variance_ratio)
        num_components = min(np.argmax(cumulative_variance >= pca_percent / 100) + 2, max_components)

    pca = PCA(n_components=num_components)
    embeddings_pca = pca.fit_transform(embeddings)
    return embeddings_pca[:, 1:], pca  # Exclude the first component

def cluster_embeddings(embeddings_pca, pca_size, prefix):
    silhouette_scores = []
    for i in range(2, min(51, len(embeddings_pca))):  # Ensure we have enough data points for clustering
        try:
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans.fit(embeddings_pca)
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(embeddings_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        except ValueError as e:
            print(f"Error in clustering with {i} clusters: {e}")
            break

    # Check if we have any valid silhouette scores
    if silhouette_scores:
        optimal_clusters = np.argmax(silhouette_scores) + 2
    else:
        raise ValueError("Could not determine optimal number of clusters. Please check the input embeddings.")

    # Perform final clustering with the determined optimal cluster number
    try:
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(embeddings_pca)
        centroids = kmeans.cluster_centers_
    except NotFittedError:
        raise ValueError("KMeans clustering failed. Please check the input data.")

    return optimal_clusters, centroids


def compute_normalized_dot_product(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def compute_adjusted_score(cosine_similarity, N, sum_cosine_similarities):
    """Compute the adjusted score based on cosine similarity."""
    adjustment_factor = min(1, np.log(N / sum_cosine_similarities))
    adjusted_score = cosine_similarity * adjustment_factor
    return adjusted_score


def calculate_adjusted_scores(task_embeddings, centroids):
    """Calculate the adjusted scores for each task against all centroids."""
    N = len(task_embeddings)
    K = len(centroids)
    adjusted_scores = np.zeros((N, K))
    
    for k in range(K):
        sum_cosine_similarities = np.sum([compute_cosine_similarity(task_embeddings[i], centroids[k]) for i in range(N)])
        
        for i in range(N):
            cosine_similarity = compute_cosine_similarity(task_embeddings[i], centroids[k])
            adjusted_scores[i, k] = compute_adjusted_score(cosine_similarity, N, sum_cosine_similarities)
    
    max_adjusted_scores = np.max(adjusted_scores, axis=1)
    return adjusted_scores, max_adjusted_scores


def save_capabilities_clusters_to_file(df, labels_col, descriptive_labels_dict, file_name):
    with open(file_name, 'w') as file:
        grouped = df.groupby(labels_col)
        for cluster_id, group in grouped:
            description = descriptive_labels_dict.get(cluster_id, "")
            file.write(f"Cluster {cluster_id}: {description}\n")
            for capability in group['Capability']:
                file.write(f"- {capability}\n")
            file.write("\n")

def generate_cluster_descriptions(df, labels_col, cluster_name):
    descriptions = {}
    for cluster_id in df[labels_col].unique():
        capabilities = df[df[labels_col] == cluster_id]['Capability'].tolist()
        word_counts = Counter(" ".join(capabilities).split())
        common_words = word_counts.most_common(10)
        descriptions[cluster_id] = f"{cluster_name} Cluster {cluster_id}: Common themes - {', '.join([word for word, count in common_words])}"
    return descriptions

def save_cluster_descriptions_to_file(descriptions, file_name):
    with open(file_name, 'w') as file:
        for cluster_id, description in descriptions.items():
            file.write(description + "\n")

def get_descriptive_labels_from_gpt(cluster_descriptions):
    descriptions = []
    for cluster_id, description in cluster_descriptions.items():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Provide a concise descriptive label for the following cluster description not more that four words: {description}"}
            ]
        )
        descriptive_label = response.choices[0].message.content.strip()
        descriptions.append((cluster_id, descriptive_label))
    return descriptions

def generate_distinct_colors(n):
    """
    Generate n visually distinct colors ensuring that no two colors are too close in hue.
    """
    if n <= 20:
        colormap = plt.cm.get_cmap('tab20', n)
        colors = [mcolors.rgb2hex(colormap(i)[:3]) for i in range(colormap.N)]
    else:
        hues = np.linspace(0, 1, n, endpoint=False)
        colors = [colorsys.hsv_to_rgb(hue, 0.7, 0.9) for hue in hues]
        colors = [mcolors.rgb2hex(color) for color in colors]
        colors = sorted(colors, key=lambda color: colorsys.rgb_to_hsv(*mcolors.hex2color(color))[0])
        mid = len(colors) // 2
        colors = list(itertools.chain.from_iterable(zip(colors[:mid], colors[mid:])))
    return colors

def visualize_clusters(embeddings, labels, title, file_name, centroids, cluster_descriptions):
    n_samples = len(embeddings)
    n_clusters = len(np.unique(labels))
    perplexity = min(30, n_samples - 1)

    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
    tsne_result = tsne.fit_transform(embeddings)

    colors = generate_distinct_colors(n_clusters)
    
    plt.figure(figsize=(20, 14))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap=mcolors.ListedColormap(colors))
    plt.colorbar(scatter, ticks=range(np.max(labels) + 1))
    plt.title(title, fontsize=16)

    centroids_perplexity = min(30, len(centroids) - 1)
    tsne_centroids = TSNE(n_components=2, verbose=1, perplexity=centroids_perplexity, n_iter=300)
    centroids_tsne = tsne_centroids.fit_transform(centroids)

    for i, centroid in enumerate(centroids_tsne):
        color = colors[i]
        plt.scatter(centroid[0], centroid[1], c=[color], marker='X', s=200, edgecolors='black', linewidths=2)
        plt.text(centroid[0], centroid[1], f'Cluster {i}', color=color, fontsize=10, 
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{file_name}', format='png', dpi=300)
    plt.show()

def combine_files_no_pca(embeddings_array, embeddings_array_openai, embeddings_array_openai_lg, embeddings_array_openai_sm, embeddings_array_e5):
    combined_df = pd.DataFrame()

    for prefix, embeddings, compute_embeddings_func, tokenizer, model in zip(
            ['bert', 'openai', 'openai_lg', 'openai_sm', 'e5'], 
            [embeddings_array, embeddings_array_openai, embeddings_array_openai_lg, embeddings_array_openai_sm, embeddings_array_e5],
            [generate_embeddings, compute_openai_embeddings, compute_openai_embeddings_lg, compute_openai_embeddings_sm, compute_e5_embeddings],
            [bert_tokenizer, None, None, None, e5_tokenizer],
            [bert_model, None, None, None, e5_model]
        ):
        task_statements_df = generate_task_embeddings(task_statements_file, embeddings, None, len(embeddings[0]), prefix, None, compute_embeddings_func, tokenizer, model)
        combined_df = pd.concat([combined_df, task_statements_df], axis=1)

    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    # combined_df.to_csv(f'{results_path}task_embeddings_cosine_normal_similarity_revised_no_pca.csv', index=False)
    combined_df.to_csv(os.path.join(results_path, 'task_embeddings_cosine_normal_similarity_revised_no_pca.csv'), index=False)

def calculate_adjusted_scores(task_embeddings, centroids):
    N = len(task_embeddings)
    K = len(centroids)
    adjusted_scores = np.zeros((N, K), dtype=np.float32)
    
    for k in range(K):
        # Ensure centroid is a numeric array
        centroid = np.asarray(centroids[k], dtype=np.float32)
        
        sum_cosine_similarities = np.sum([
            compute_cosine_similarity(task_embeddings[i], centroid) for i in range(N)
        ])
        
        for i in range(N):
            cosine_similarity = compute_cosine_similarity(task_embeddings[i], centroid)
            adjusted_scores[i, k] = compute_adjusted_score(cosine_similarity, N, sum_cosine_similarities)
    
    max_adjusted_scores = np.max(adjusted_scores, axis=1)
    return adjusted_scores, max_adjusted_scores




def compute_cluster_description_task_similarity(clusters_file, task_statements_file, compute_embeddings_func, tokenizer=None, model=None, prefix=""):
    cluster_descriptions = {}
    with open(clusters_file, 'r', encoding='utf-8') as file:
        current_cluster = None
        description_lines = []
        for line in file:
            if line.startswith('Cluster'):
                if current_cluster is not None:
                    cluster_descriptions[current_cluster] = " ".join(description_lines)
                current_cluster = line.split(":")[0].strip()
                description_lines = []
            else:
                description_lines.append(line.strip())
        if current_cluster is not None:
            cluster_descriptions[current_cluster] = " ".join(description_lines)

    # Check if compute_embeddings_func is an OpenAI function
    def is_openai_func(func):
        return func in [compute_openai_embeddings, compute_openai_embeddings_lg, compute_openai_embeddings_sm]

    # Compute embeddings for cluster descriptions
    cluster_description_embeddings = {
        cluster_id: (compute_embeddings_func(preprocess_text(description))
                     if is_openai_func(compute_embeddings_func)
                     else compute_embeddings_func(preprocess_text(description), tokenizer, model))
        for cluster_id, description in cluster_descriptions.items()
    }

    # Load task statements and compute embeddings
    task_statements_df = pd.read_excel(task_statements_file)
    task_embeddings = task_statements_df['Task'].apply(
        lambda x: (compute_embeddings_func(preprocess_text(x))
                   if is_openai_func(compute_embeddings_func)
                   else compute_embeddings_func(preprocess_text(x), tokenizer, model))
    )
    task_embeddings_array = np.vstack(task_embeddings)


# Updating the logic 


   # Compute cosine similarities and normalized dot products
    N = len(task_embeddings_array)
    for i, (cluster_id, cluster_embedding) in enumerate(cluster_description_embeddings.items()):
        cosine_similarities = [compute_cosine_similarity(task_emb, cluster_embedding) for task_emb in task_embeddings_array]
        normalized_dot_products = [compute_normalized_dot_product(task_emb, cluster_embedding) for task_emb in task_embeddings_array]
        sum_cosine_similarities = sum(cosine_similarities)
        adjustment_factor = min(1, np.log(N / sum_cosine_similarities)) if sum_cosine_similarities > 0 else 0
        adjusted_scores = [similarity * adjustment_factor for similarity in cosine_similarities]

        # Populate the DataFrame with calculated values
        task_statements_df[f'Cluster_{prefix}_{cluster_id}_Cosine'] = cosine_similarities
        task_statements_df[f'Cluster_{prefix}_{cluster_id}_NormDotProduct'] = normalized_dot_products
        task_statements_df[f'Cluster_{prefix}_{cluster_id}_AdjustedScore'] = adjusted_scores

    # Calculate and add maximum and average values for each metric
    metrics = ['Cosine', 'NormDotProduct', 'AdjustedScore']
    for metric in metrics:
        columns = [f'Cluster_{prefix}_{cluster_id}_{metric}' for cluster_id in cluster_description_embeddings.keys()]
        task_statements_df[f'Max_{metric}_{prefix}'] = task_statements_df[columns].max(axis=1)
        task_statements_df[f'Avg_{metric}_{prefix}'] = task_statements_df[columns].mean(axis=1)

    # task_statements_df.to_csv(f'{results_path}task_embeddings_cosine_normal_adjusted_similarity_revised_{prefix}.csv', index=False)
    task_statements_df.to_csv(os.path.join(results_path, f'task_embeddings_cosine_normal_adjusted_similarity_revised_{prefix}.csv'), index=False)
    
    return task_statements_df






  

def perform_clustering_and_similarity(embeddings, filename_prefix, compute_embeddings_func, tokenizer=None, model=None):
    optimal_clusters, centroids = cluster_embeddings(embeddings, len(embeddings[0]), filename_prefix)
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # np.save(f'{results_path}labels_{filename_prefix}.npy', labels)
    # np.save(f'{results_path}centroids_{filename_prefix}.npy', centroids)
    np.save(os.path.join(results_path, f'labels_{filename_prefix}.npy'), labels)
    np.save(os.path.join(results_path, f'centroids_{filename_prefix}.npy'), centroids)
    task_statements_df = pd.read_excel(task_statements_file)

    task_embeddings = task_statements_df['Task'].apply(lambda x: compute_embeddings_func(preprocess_text(x), tokenizer, model) if tokenizer and model else compute_embeddings_func(preprocess_text(x)))
    task_embeddings_array = np.vstack(task_embeddings)

    adjusted_scores, max_adjusted_scores = calculate_adjusted_scores(task_embeddings_array, centroids)

    for i, centroid in enumerate(centroids):
        task_statements_df[f'Cluster_{filename_prefix}_{i}_Cosine'] = [
            compute_cosine_similarity(task_emb, centroid) for task_emb in task_embeddings_array]
        task_statements_df[f'Cluster_{filename_prefix}_{i}_NormDotProduct'] = [
            compute_normalized_dot_product(task_emb, centroid) for task_emb in task_embeddings_array
        ]
        task_statements_df[f'Cluster_{filename_prefix}_{i}_AdjustedScore'] = adjusted_scores[:, i]

    task_statements_df[f'Max_Cosine_{filename_prefix}'] = task_statements_df[[f'Cluster_{filename_prefix}_{i}_Cosine' for i in range(len(centroids))]].max(axis=1)
    task_statements_df[f'Avg_Cosine_{filename_prefix}'] = task_statements_df[[f'Cluster_{filename_prefix}_{i}_Cosine' for i in range(len(centroids))]].mean(axis=1)
    task_statements_df[f'Max_NormDotProduct_{filename_prefix}'] = task_statements_df[[f'Cluster_{filename_prefix}_{i}_NormDotProduct' for i in range(len(centroids))]].max(axis=1)
    task_statements_df[f'Avg_NormDotProduct_{filename_prefix}'] = task_statements_df[[f'Cluster_{filename_prefix}_{i}_NormDotProduct' for i in range(len(centroids))]].mean(axis=1)
    task_statements_df[f'Max_AdjustedScore_{filename_prefix}'] = max_adjusted_scores

    # task_statements_df.to_csv(f'{results_path}task_embeddings_cosine_normal_adjusted_similarity_revised_{filename_prefix}.csv', index=False)
    task_statements_df.to_csv(os.path.join(results_path, f'task_embeddings_cosine_normal_adjusted_similarity_revised_{filename_prefix}.csv'), index=False)
    
    return task_statements_df

# Save capabilities with unique IDs and cluster labels
def perform_clustering_and_save(embeddings, model_name, pca_size, pca_percent, compute_embeddings_func, tokenizer=None, model=None, llm_capabilities=None):
    # Perform clustering
    optimal_clusters, _ = cluster_embeddings(embeddings, len(embeddings[0]), model_name)
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Create DataFrame including ai_unique_id and cluster labels
    capabilities_df = pd.DataFrame({
        'Capability': [cap['text'] for cap in llm_capabilities],
        'ai_unique_id': [cap['ai_unique_id'] for cap in llm_capabilities],
        'Cluster_Label': labels
    })

    # Save the clustered capabilities to a CSV file
    clusters_file = os.path.join(results_path, f'capabilities_clusters_{model_name}_{pca_size if pca_size > 0 else pca_percent}.csv')
    capabilities_df.to_csv(clusters_file, index=False)
    
    # Return the file path for later use
    return clusters_file



def compute_granular_similarity(task_statements_df, clusters_file, compute_embeddings_func, tokenizer=None, model=None):
    clusters_df = pd.read_csv(clusters_file)

    for index, task_row in task_statements_df.iterrows():
        task_embedding = task_row['Task_Embedding']
        cluster_id = task_row['Cluster_ID']
        
        # Filter capabilities within the identified cluster
        capabilities_in_cluster = clusters_df[clusters_df['Cluster_Label'] == cluster_id]
        
        # Compute similarity with each capability in the cluster
        similarities = []
        for _, cap_row in capabilities_in_cluster.iterrows():
            cap_embedding = load_embedding_by_id(cap_row['ai_unique_id'])  # Retrieve from saved embeddings
            similarity = compute_cosine_similarity(task_embedding, cap_embedding)
            similarities.append((cap_row['ai_unique_id'], similarity))
        
        # Get max similarity and associated ai_unique_id
        max_ai_unique_id, max_similarity = max(similarities, key=lambda x: x[1])
        task_statements_df.at[index, 'Granular_Cosine_Similarity'] = max_similarity
        task_statements_df.at[index, 'Max_AI_Capability_ID'] = max_ai_unique_id
    
    return task_statements_df


def combine_results(models_dfs,pca_percent):
    combined_df = pd.concat(models_dfs, axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]  # Remove duplicate columns
    # combined_df.to_csv(f'{results_path}task_embeddings_cosine_normal_adjusted_similarity_revised_pca_{pca_percent}.csv', index=False)
    combined_df.to_csv(os.path.join(results_path, f'task_embeddings_cosine_normal_adjusted_similarity_revised_pca_{pca_percent}.csv'), index=False)

    return combined_df


# Dictionary for models and embedding functions
embeddings_dict = {
    # 'bert': (generate_embeddings, bert_tokenizer, bert_model),
    # 'openai': (compute_openai_embeddings, None, None),
    'openai_lg': (compute_openai_embeddings_lg, None, None),
    # 'openai_sm': (compute_openai_embeddings_sm, None, None),
    'e5': (compute_e5_embeddings, e5_tokenizer, e5_model)
}

# Load task statements as DataFrame
def load_task_statements(file_path):
    return pd.read_excel(file_path)

# Helper function to load embeddings by ai_unique_id
def load_embedding_by_id(ai_unique_id, embeddings_dict):
    # Assuming embeddings are stored in a dictionary with ai_unique_id as the key
    return embeddings_dict.get(ai_unique_id)

#     return task_statements_df, task_embeddings_pca
def generate_task_embeddings(task_statements_file, clusters_file, prefix, compute_embeddings_func, tokenizer=None, model=None):
    # Load task statements
    task_statements_df = pd.read_excel(task_statements_file)
    
    # Check if the embedding function requires only one argument (e.g., OpenAI functions)
    def is_single_arg_func(func):
        return func in [compute_openai_embeddings, compute_openai_embeddings_lg, compute_openai_embeddings_sm]

    # Generate embeddings for each task statement
    task_embeddings = task_statements_df['Task'].apply(
        lambda x: compute_embeddings_func(preprocess_text(x)) if is_single_arg_func(compute_embeddings_func)
        else compute_embeddings_func(preprocess_text(x), tokenizer, model)
    )
    task_embeddings = np.array([np.array(embed, dtype=np.float32) for embed in task_embeddings])

    # Load the clustered capabilities from the file
    clusters_df = pd.read_csv(clusters_file)
    
    closest_capabilities = []
    closest_clusters = []
    max_similarities_cluster = []
    max_similarities_capability = []

    # Step 1: Compute Task-to-Cluster Similarity
    cluster_embeddings = {}
    for cluster_id in clusters_df['Cluster_Label'].unique():
        # Get all capabilities in the cluster and calculate the average embedding as the cluster embedding
        capabilities_in_cluster = clusters_df[clusters_df['Cluster_Label'] == cluster_id]
        capability_embeddings = np.array([
            compute_embeddings_func(preprocess_text(cap['Capability'])) if is_single_arg_func(compute_embeddings_func)
            else compute_embeddings_func(preprocess_text(cap['Capability']), tokenizer, model)
            for _, cap in capabilities_in_cluster.iterrows()
        ])
        cluster_embeddings[cluster_id] = capability_embeddings.mean(axis=0)
    
    for task_embedding in task_embeddings:
        # Step 2: Find the Most Similar Cluster
        cluster_similarities = {cluster_id: compute_cosine_similarity(task_embedding, cluster_embedding) for cluster_id, cluster_embedding in cluster_embeddings.items()}
        most_similar_cluster = max(cluster_similarities, key=cluster_similarities.get)
        max_similarity_cluster = cluster_similarities[most_similar_cluster]
        
        # Step 3: Compute Task-to-Capability Similarity within the Closest Cluster
        capabilities_in_most_similar_cluster = clusters_df[clusters_df['Cluster_Label'] == most_similar_cluster]
        capability_similarities = []
        
        for _, capability in capabilities_in_most_similar_cluster.iterrows():
            capability_embedding = compute_embeddings_func(preprocess_text(capability['Capability'])) if is_single_arg_func(compute_embeddings_func) else compute_embeddings_func(preprocess_text(capability['Capability']), tokenizer, model)
            similarity = compute_cosine_similarity(task_embedding, capability_embedding)
            capability_similarities.append((capability['ai_unique_id'], similarity))
        
        # Identify the capability with the highest similarity within the chosen cluster
        max_ai_unique_id, max_similarity_capability = max(capability_similarities, key=lambda x: x[1])

        # Store the results
        closest_clusters.append(most_similar_cluster)
        max_similarities_cluster.append(max_similarity_cluster)
        closest_capabilities.append(max_ai_unique_id)
        max_similarities_capability.append(max_similarity_capability)

    # Add results to the DataFrame
    task_statements_df[f'Most_Similar_Cluster_ID_{prefix}'] = closest_clusters
    task_statements_df[f'Max_Similarity_Cluster_{prefix}'] = max_similarities_cluster
    task_statements_df[f'Most_Similar_Capability_ID_{prefix}'] = closest_capabilities
    task_statements_df[f'Max_Similarity_Capability_{prefix}'] = max_similarities_capability

    # Save the updated DataFrame
    output_file = os.path.join(results_path, f'task_embeddings_cosine_normal_adjusted_similarity_revised_{prefix}.csv')
    task_statements_df.to_csv(output_file, index=False)
    
    return task_statements_df



# Main function to control the entire process
def main(pca_size=0, pca_percent=0):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    llm_capabilities = load_and_preprocess_capabilities(llm_capabilities_file)
    
    # Step 1: Generate and save embeddings for each capability
    embeddings_data = {}  # Dictionary to store embeddings with ai_unique_id for retrieval
    for cap in llm_capabilities:
        text = preprocess_text(cap['text'])
        ai_unique_id = cap['ai_unique_id']
        embeddings_data[ai_unique_id] = {
            # 'bert': generate_embeddings(text, bert_tokenizer, bert_model),
            # 'openai': compute_openai_embeddings(text),
            'openai_lg': compute_openai_embeddings_lg(text),
            # 'openai_sm': compute_openai_embeddings_sm(text),
            'e5': compute_e5_embeddings(text, e5_tokenizer, e5_model)
        }

    models_dfs = []
    for model_name, (compute_func, tokenizer, model) in embeddings_dict.items():
        embeddings = [data[model_name] for data in embeddings_data.values()]
        embeddings_array = np.vstack(embeddings)

        # Save the original embeddings (before PCA)
        original_embeddings_file = os.path.join(results_path, f'embeddings_original_{model_name}.npy')
        np.save(original_embeddings_file, embeddings_array)
        print(f"Saved original embeddings for {model_name} to {original_embeddings_file}")

        # Step 2: Save clustered capabilities without centroids
        clusters_file = perform_clustering_and_save(
            embeddings_array, model_name, pca_size, pca_percent, compute_func, tokenizer, model, llm_capabilities
        )

        # Step 3: Generate task embeddings and calculate similarities
        task_statements_df = generate_task_embeddings(
            task_statements_file, clusters_file, model_name, compute_func, tokenizer, model
        )

        # Add the DataFrame with results for this model to the list
        models_dfs.append(task_statements_df)
    
    # Step 4: Combine results across models into a single DataFrame
    final_df = combine_results(models_dfs, pca_percent)

    # Save the final combined results
    final_output_file = os.path.join(results_path, f'task_embeddings_cosine_normal_adjusted_similarity_revised_pca_{pca_percent}.csv')
    final_df.to_csv(final_output_file, index=False)

    print("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PCA files and generate embeddings.")
    parser.add_argument('--pca_size', type=int, default=0, help="PCA size to be used for the embeddings.")
    parser.add_argument('--pca_percent', type=float, default=0, help="PCA percentage to be used for the embeddings.")
    
    args = parser.parse_args()
    main(args.pca_size, args.pca_percent)













