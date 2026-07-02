import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Cấu hình đường dẫn xuất ảnh trực tiếp vào thư mục Artifact
ARTIFACT_DIR = r"C:\Users\DELL\.gemini\antigravity\brain\99801628-911d-4d92-9f31-e631aed997f1"

# Cấu hình font chữ tiếng Việt cho matplotlib trên Windows
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_parse_data(filepath="chat_logs_dataset.csv"):
    df = pd.read_csv(filepath)
    df.dropna(subset=['Cuộc trò chuyện'], inplace=True)
    df.drop_duplicates(subset=['Cuộc trò chuyện'], inplace=True)
    
    parsed_data = []
    customer_texts = []
    
    for text in df['Cuộc trò chuyện']:
        lines = text.split('\n')
        conv_turns = len(lines)
        customer_len = []
        page_len = []
        
        for line in lines:
            if line.startswith("Khách hàng:"):
                msg = line.replace("Khách hàng:", "", 1).strip()
                if msg:
                    customer_len.append(len(msg))
                    customer_texts.append(msg)
            elif line.startswith("Page:"):
                msg = line.replace("Page:", "", 1).strip()
                if msg:
                    page_len.append(len(msg))
                    
        parsed_data.append({
            "Turns": conv_turns,
            "Avg_Customer_Len": sum(customer_len)/len(customer_len) if customer_len else 0,
            "Avg_Page_Len": sum(page_len)/len(page_len) if page_len else 0
        })
        
    return pd.DataFrame(parsed_data), customer_texts

def plot_message_lengths(stats_df):
    plt.figure(figsize=(10, 6))
    data_to_plot = pd.melt(stats_df[['Avg_Customer_Len', 'Avg_Page_Len']], 
                           var_name='Sender', value_name='Length')
    # Thay đổi label cho đẹp
    data_to_plot['Sender'] = data_to_plot['Sender'].map({'Avg_Customer_Len': 'Khách hàng', 'Avg_Page_Len': 'Page'})
    
    sns.boxplot(x='Sender', y='Length', data=data_to_plot, palette='Set2')
    plt.title('Phân phối Độ dài Trung bình Tin nhắn (Số ký tự)', fontsize=14, fontweight='bold')
    plt.ylabel('Số ký tự')
    plt.ylim(0, 300) # Giới hạn trục Y để nhìn rõ boxplot, cắt bỏ outliers quá dài
    
    out_path = os.path.join(ARTIFACT_DIR, 'boxplot_length.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_wordcloud(customer_texts):
    text = " ".join(customer_texts).lower()
    # Loại bỏ các từ vô nghĩa và các xưng hô thông thường
    stopwords = set([
        "khách", "hàng", "page", "dạ", "chào", "vâng", "anh", "chị", "em", "bạn", 
        "của", "là", "thì", "mà", "có", "không", "cho", "mình", "ở", "nào", "đang", 
        "đã", "được", "rồi", "này", "ạ", "ơi", "nhé", "nha", "để", "xin", "cảm ơn"
    ])
    
    # WordCloud mặc định có thể lỗi font tiếng Việt, cần truyền đường dẫn font Arial
    font_path = "C:/Windows/Fonts/arial.ttf"
    if not os.path.exists(font_path):
        font_path = None # Dùng mặc định nếu không tìm thấy
        
    wc = WordCloud(width=800, height=400, background_color='white', 
                   colormap='viridis', max_words=100, font_path=font_path,
                   stopwords=stopwords).generate(text)
                   
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Đám Mây Từ Vựng (Word Cloud) - Tin nhắn Khách Hàng', fontsize=16, fontweight='bold')
    
    out_path = os.path.join(ARTIFACT_DIR, 'wordcloud.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_top_bigrams(customer_texts):
    # Dùng CountVectorizer để tìm Bigrams (cụm 2 từ)
    stopwords = [
        "dạ", "chào", "vâng", "anh", "chị", "em", "bạn", "của", "là", "thì", "mà", "có", 
        "không", "cho", "mình", "ở", "nào", "đang", "đã", "được", "rồi", "này", "ạ", "ơi",
        "nhé", "nha", "để", "xin", "cảm", "ơn", "cảm ơn", "mình hỏi", "cho mình", "anh chị",
        "dạ chào", "chào page"
    ]
    
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=stopwords, max_features=15)
    X = vectorizer.fit_transform(customer_texts)
    
    # Tính tổng số lần xuất hiện
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    df_freq = pd.DataFrame(words_freq, columns=['Bigram', 'Count'])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Count', y='Bigram', data=df_freq, palette='magma')
    plt.title('Top 15 Cụm từ (Bigrams) Khách hàng hay dùng nhất', fontsize=14, fontweight='bold')
    plt.xlabel('Tần suất xuất hiện')
    plt.ylabel('Cụm từ')
    
    out_path = os.path.join(ARTIFACT_DIR, 'top_bigrams.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    print("Loading data...")
    stats_df, customer_texts = load_and_parse_data("chat_logs_dataset.csv")
    
    print("Plotting Message Lengths Boxplot...")
    plot_message_lengths(stats_df)
    
    print("Plotting Word Cloud...")
    plot_wordcloud(customer_texts)
    
    print("Plotting Top Bigrams...")
    plot_top_bigrams(customer_texts)
    
    print("All visual EDA tasks completed!")

if __name__ == "__main__":
    main()
