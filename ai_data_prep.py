import pandas as pd
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def load_and_clean_data(input_file="chat_logs_dataset.csv"):
    print(f"[*] Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    original_count = len(df)
    
    # Drop NAs
    df.dropna(subset=['Cuộc trò chuyện'], inplace=True)
    
    # Deduplicate
    df.drop_duplicates(subset=['Cuộc trò chuyện'], inplace=True)
    
    # Filter by length (e.g. drop very short interactions like just "chào" without follow-ups)
    df['char_length'] = df['Cuộc trò chuyện'].apply(len)
    df = df[df['char_length'] > 20].copy()
    
    print(f"[*] Data Cleaned: {len(df)} remaining from {original_count} original entries.")
    return df

def cluster_intents(df, num_clusters=10, output_file="intent_clusters.csv"):
    print("[*] Running TF-IDF and K-Means clustering for Intent Discovery...")
    
    # Basic Vietnamese stopwords to ignore
    vi_stopwords = [
        "là", "và", "của", "thì", "mà", "có", "không", "ở", "với", "để",
        "cho", "này", "cũng", "được", "trong", "những", "các", "một", "như", "hay"
    ]
    
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words=vi_stopwords, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['Cuộc trò chuyện'])
    
    # Use MiniBatchKMeans for faster execution on 30k+ rows
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    df['Intent_Cluster'] = kmeans.fit_predict(X)
    
    # Extract top keywords per cluster
    print("[*] Extracting top keywords per cluster...")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_labels = {}
    for i in range(num_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :5]]
        cluster_labels[i] = " | ".join(top_terms)
        
    df['Cluster_Keywords'] = df['Intent_Cluster'].map(cluster_labels)
    
    # Save a sample of clustered data
    cluster_summary = df[['STT', 'Intent_Cluster', 'Cluster_Keywords', 'Cuộc trò chuyện']].copy()
    cluster_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[+] Clustering complete. Saved cluster assignments to {output_file}")
    
    for i in range(num_clusters):
        count = len(df[df['Intent_Cluster'] == i])
        print(f"    - Cluster {i} ({count} items): {cluster_labels[i]}")
        
    return df

def parse_conversation_to_messages(text: str):
    """
    Splits the 'Cuộc trò chuyện' text into a list of OpenAI format messages.
    """
    lines = text.split('\n')
    messages = [
        {"role": "system", "content": "Bạn là một nhân viên chăm sóc khách hàng nhiệt tình và chuyên nghiệp của Fanpage. Nhiệm vụ của bạn là tư vấn và giải đáp thắc mắc của khách hàng một cách thân thiện."}
    ]
    
    current_role = None
    current_content = []
    
    for line in lines:
        if line.startswith("Khách hàng:"):
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role = "user"
            current_content = [line.replace("Khách hàng:", "", 1).strip()]
        elif line.startswith("Page:"):
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role = "assistant"
            current_content = [line.replace("Page:", "", 1).strip()]
        else:
            # Multi-line message continuation
            current_content.append(line.strip())
            
    # Append the last message
    if current_role and current_content:
        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
        
    # Ensure it ends with assistant for training efficiency
    # If the last message is from user, it doesn't give the model a target to learn.
    # We only yield conversations where the assistant actually replied.
    if messages[-1]['role'] == 'assistant':
        return messages
    return None

def format_for_llm(df, openai_out="train_openai.jsonl", alpaca_out="train_alpaca.jsonl"):
    print("[*] Formatting data to JSONL for LLM Fine-Tuning...")
    
    openai_data = []
    alpaca_data = []
    
    for text in tqdm(df['Cuộc trò chuyện'], desc="Converting formats"):
        messages = parse_conversation_to_messages(text)
        if messages and len(messages) >= 3: # System + at least 1 user + 1 assistant
            
            # OpenAI Format
            openai_data.append({"messages": messages})
            
            # Alpaca Format (Simplified: Input = all prior context, Output = last assistant message)
            # We will just take the first user message as input, and the assistant's first response as output for single-turn Alpaca
            # For multi-turn, OpenAI format is much better. We provide a basic single-turn Alpaca mapping.
            user_msg = messages[1]['content']
            asst_msg = messages[2]['content']
            alpaca_data.append({
                "instruction": "Bạn là nhân viên tư vấn khách hàng. Hãy trả lời câu hỏi sau của khách:",
                "input": user_msg,
                "output": asst_msg
            })
            
    # Save OpenAI
    with open(openai_out, 'w', encoding='utf-8') as f:
        for item in openai_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # Save Alpaca
    with open(alpaca_out, 'w', encoding='utf-8') as f:
        for item in alpaca_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"[+] Saved {len(openai_data)} examples to {openai_out}")
    print(f"[+] Saved {len(alpaca_data)} examples to {alpaca_out}")

def main():
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    if not os.path.exists("chat_logs_dataset.csv"):
        print("[-] Error: chat_logs_dataset.csv not found!")
        return
        
    df = load_and_clean_data("chat_logs_dataset.csv")
    df = cluster_intents(df, num_clusters=8, output_file="intent_clusters.csv")
    format_for_llm(df)
    
if __name__ == "__main__":
    main()
