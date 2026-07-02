import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import os
import sys

def parse_first_turn(text):
    """
    Trích xuất lượt hỏi đáp đầu tiên trong đoạn chat:
    - User Query: Lời khách hàng đầu tiên.
    - Page Response: Lời tư vấn viên phản hồi lại ngay sau đó.
    """
    lines = text.split('\n')
    user_query = []
    page_response = []
    
    state = "START" # START -> USER -> PAGE -> DONE
    
    for line in lines:
        if line.startswith("Khách hàng:"):
            if state == "START" or state == "USER":
                state = "USER"
                user_query.append(line.replace("Khách hàng:", "", 1).strip())
            elif state == "PAGE":
                state = "DONE" # Chỉ lấy turn đầu tiên
                break
        elif line.startswith("Page:"):
            if state == "USER" or state == "PAGE":
                state = "PAGE"
                page_response.append(line.replace("Page:", "", 1).strip())
        else:
            # Multi-line
            if state == "USER":
                user_query.append(line.strip())
            elif state == "PAGE":
                page_response.append(line.strip())
                
    uq = " ".join(user_query).strip()
    pr = " ".join(page_response).strip()
    
    if uq and pr:
        return uq, pr
    return None, None

def score_response(response_text):
    """
    Chấm điểm chất lượng câu trả lời của Page.
    - Base score: Độ dài (càng dài càng tốt, nhưng cap ở 500 ký tự).
    - Bonus: Chứa các từ lịch sự (dạ, vâng, chào, cảm ơn).
    - Penalty: Quá ngắn (< 20 ký tự), hoặc chứa từ khóa spam/lười (inbox, ib).
    """
    score = 0
    lower_res = response_text.lower()
    
    # 1. Base score by length (max 50 points)
    length = len(response_text)
    score += min(length / 10, 50)
    
    # 2. Politeness bonus (max 30 points)
    polite_words = ['dạ', 'vâng', 'chào', 'cảm ơn', 'anh', 'chị', 'mình']
    for word in polite_words:
        if word in lower_res:
            score += 5
            
    # 3. Informative bonus (Numbers, prices often indicate specific info)
    numbers = len(re.findall(r'\d+', response_text))
    score += min(numbers * 2, 20)
    
    # 4. Penalties
    if length < 20:
        score -= 50
    if 'inbox' in lower_res or 'ib' in lower_res:
        score -= 20
        
    return score

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("[*] Đang tải dữ liệu gốc...")
    if not os.path.exists("chat_logs_dataset.csv"):
        print("[-] Không tìm thấy file chat_logs_dataset.csv")
        return
        
    df_raw = pd.read_csv("chat_logs_dataset.csv")
    df_raw.dropna(subset=['Cuộc trò chuyện'], inplace=True)
    df_raw.drop_duplicates(subset=['Cuộc trò chuyện'], inplace=True)
    
    print("[*] Đang trích xuất Lượt chat đầu tiên (First Turn Extraction)...")
    parsed_rows = []
    for idx, row in df_raw.iterrows():
        uq, pr = parse_first_turn(row['Cuộc trò chuyện'])
        if uq and pr and len(uq) > 5: # Bỏ qua các câu hỏi quá ngắn như "ơ", "?"
            parsed_rows.append({
                "STT": row['STT'],
                "Customer_Query": uq,
                "Page_Response": pr,
                "Quality_Score": score_response(pr)
            })
            
    df = pd.DataFrame(parsed_rows)
    print(f"[-] Đã lọc ra {len(df)} lượt hỏi-đáp hợp lệ.")
    
    print("[*] Đang phân cụm dữ liệu thành 150 kịch bản siêu nhỏ (Micro-Intents)...")
    vi_stopwords = [
        "dạ", "chào", "anh", "chị", "em", "bạn", "page", "ơi", "cho", "hỏi", "mình",
        "là", "và", "của", "thì", "mà", "có", "không", "ở", "với", "để", "này", "ạ"
    ]
    
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=3, stop_words=vi_stopwords, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['Customer_Query'])
    
    NUM_CLUSTERS = 150
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init="auto")
    df['Intent_ID'] = kmeans.fit_predict(X)
    
    # Extract keywords for clusters
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for i in range(NUM_CLUSTERS):
        top_terms = [terms[ind] for ind in order_centroids[i, :4]]
        cluster_keywords[i] = " | ".join(top_terms)
        
    df['Topic_Keywords'] = df['Intent_ID'].map(cluster_keywords)
    
    print("[*] Đang trích xuất Golden Response (câu trả lời chất lượng nhất) cho từng kịch bản...")
    # Lấy index của row có Quality_Score cao nhất trong mỗi Intent_ID
    golden_indices = df.groupby('Intent_ID')['Quality_Score'].idxmax()
    df_golden = df.loc[golden_indices].copy()
    
    # Sắp xếp và dọn dẹp format
    df_golden.sort_values('Intent_ID', inplace=True)
    df_golden = df_golden[['Intent_ID', 'Topic_Keywords', 'Customer_Query', 'Page_Response', 'Quality_Score']]
    df_golden.rename(columns={'Page_Response': 'Golden_Response'}, inplace=True)
    
    output_file = "chatbot_test_suite.csv"
    df_golden.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"[+] Xong! Đã tạo thành công {len(df_golden)} kịch bản mẫu.")
    print(f"[+] Dữ liệu được lưu tại: {output_file}")
    
if __name__ == "__main__":
    main()
