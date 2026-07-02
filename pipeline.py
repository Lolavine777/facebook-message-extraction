import os
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def load_config():
    load_dotenv()
    page_id = os.getenv("PAGE_ID")
    page_access_token = os.getenv("PAGE_ACCESS_TOKEN")
    
    missing = []
    if not page_id:
        missing.append("PAGE_ID")
    if not page_access_token:
        missing.append("PAGE_ACCESS_TOKEN")
        
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
    return {
        "PAGE_ID": page_id,
        "PAGE_ACCESS_TOKEN": page_access_token,
        "GRAPH_API_VERSION": os.getenv("GRAPH_API_VERSION", "v25.0")
    }

def clean_and_mask_pii(text: str) -> str:
    if not text:
        return text
    
    # Mask Email
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[EMAIL]', text)
    
    # Mask Phone (starts with 0 or 84, followed by 9 digits)
    # also matching potential spaces or dots is a bonus but let's keep it simple first
    phone_pattern = r'\b(0|84)\d{9}\b'
    text = re.sub(phone_pattern, '[SĐT]', text)
    
    return text

class RateLimitException(Exception):
    pass

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RateLimitException)
)
def _fetch_url(url: str) -> dict:
    response = requests.get(url)
    if response.status_code == 429:
        raise RateLimitException("Rate limit hit")
    response.raise_for_status()
    return response.json()

def fetch_conversations(config: dict) -> list:
    url = f"https://graph.facebook.com/{config['GRAPH_API_VERSION']}/{config['PAGE_ID']}/conversations?fields=messages{{message,from,created_time}}&access_token={config['PAGE_ACCESS_TOKEN']}"
    
    all_conversations = []
    
    while url:
        data = _fetch_url(url)
        if "data" in data:
            all_conversations.extend(data["data"])
            
        # Pagination
        url = data.get("paging", {}).get("next")
        
    return all_conversations

def process_conversations(raw_data: list, config: dict) -> list:
    processed = []
    stt = 1
    page_id = config.get("PAGE_ID")
    
    for conv in raw_data:
        messages_data = conv.get("messages", {}).get("data", [])
        if not messages_data:
            continue
            
        # 1. Reverse chronological sort (Meta returns newest first)
        messages_data.reverse()
        
        valid_messages = []
        for msg in messages_data:
            text = msg.get("message", "").strip()
            
            # 2. Filter empty messages
            if not text:
                continue
                
            # 3. PII Masking
            text = clean_and_mask_pii(text)
            
            # 4. Map sender_id to label
            sender_id = msg.get("from", {}).get("id")
            label = "Page" if str(sender_id) == str(page_id) else "Khách hàng"
            
            valid_messages.append(f"{label}: {text}")
            
        # 5. Quality Filter: Drop if < 2 valid interactions
        if len(valid_messages) < 2:
            continue
            
        # 6. Concatenate
        conversation_text = "\n".join(valid_messages)
        
        processed.append({
            "STT": stt,
            "Cuộc trò chuyện": conversation_text
        })
        stt += 1
        
    return processed

def export_to_csv(data: list, file_path: str):
    if not data:
        print("No data to export.")
        return
        
    df = pd.DataFrame(data)
    # Ensure columns match requirements exactly
    if "STT" in df.columns and "Cuộc trò chuyện" in df.columns:
        df = df[["STT", "Cuộc trò chuyện"]]
        
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

