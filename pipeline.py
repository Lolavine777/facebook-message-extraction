import os
import re
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

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

def remove_sales_templates(text: str) -> str:
    if not text:
        return text
    
    # Simple regex to catch templates with multiple "CS" or "Cơ sở" blocks
    template_pattern = r'(?:CS\d+|Cơ sở \d+)[\s\S]*'
    if re.search(template_pattern, text, re.IGNORECASE):
        text = re.sub(template_pattern, '[THÔNG_TIN_CƠ_SỞ]', text, flags=re.IGNORECASE)
        
    return text

def extract_phone_number(text: str) -> str:
    if not text:
        return ""
    # Find first matching phone number
    phone_pattern = r'\b(0|84)\d{9}\b'
    match = re.search(phone_pattern, text)
    if match:
        return match.group(0)
    return ""

class RateLimitException(Exception):
    pass

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitException, requests.exceptions.HTTPError))
)
def _fetch_url(session: requests.Session, url: str) -> dict:
    response = session.get(url)
    if response.status_code == 429:
        raise RateLimitException("Rate limit hit")
    if response.status_code >= 500:
        raise requests.exceptions.HTTPError(f"{response.status_code} Server Error", response=response)
    response.raise_for_status()
    return response.json()

def fetch_conversations(config: dict) -> list:
    url = f"https://graph.facebook.com/{config['GRAPH_API_VERSION']}/{config['PAGE_ID']}/conversations?fields=updated_time,messages{{message,from,created_time}}&access_token={config['PAGE_ACCESS_TOKEN']}"
    
    all_conversations = []
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=2*365)
    
    with requests.Session() as session:
        with tqdm(desc="Fetching API pages", unit=" page") as pbar:
            while url:
                try:
                    data = _fetch_url(session, url)
                    if "data" in data:
                        for conv in data["data"]:
                            updated_time_str = conv.get("updated_time")
                            if updated_time_str:
                                # Example: 2026-07-06T10:00:00+0000
                                updated_time = datetime.strptime(updated_time_str, "%Y-%m-%dT%H:%M:%S%z")
                                if updated_time < cutoff_date:
                                    print("\nReached conversations older than 2 years. Stopping fetch.")
                                    return all_conversations
                            
                            all_conversations.append(conv)
                        
                    # Pagination
                    url = data.get("paging", {}).get("next")
                    pbar.update(1)
                except (requests.exceptions.RequestException, RateLimitException, RetryError) as e:
                    print(f"\nWarning: Failed to fetch data (stopping pagination). Error: {str(e)}")
                    break
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt detected! Stopping fetch safely...")
                    break
        
    return all_conversations

def process_conversations(raw_data: list, config: dict) -> list:
    processed = []
    stt = 1
    page_id = config.get("PAGE_ID")
    
    for conv in tqdm(raw_data, desc="Processing conversations", unit=" conv"):
        messages_data = conv.get("messages", {}).get("data", [])
        if not messages_data:
            continue
            
        messages_data.reverse()
        
        valid_messages = []
        customer_name = ""
        customer_id = ""
        timestamp = conv.get("updated_time", "")
        phone_number = ""
        
        for msg in messages_data:
            text = msg.get("message", "").strip()
            
            if not text:
                continue
                
            sender = msg.get("from", {})
            sender_id = sender.get("id")
            
            if str(sender_id) == str(page_id):
                label = "Page"
            else:
                label = "Khách hàng"
                # Extract customer info
                if not customer_id:
                    customer_id = sender_id
                    customer_name = sender.get("name", "")
            
            # Extract phone before template reduction
            if not phone_number and label == "Khách hàng":
                phone = extract_phone_number(text)
                if phone:
                    phone_number = phone
            
            # Reduce template
            text = remove_sales_templates(text)
            
            valid_messages.append(f"{label}: {text}")
            
        if len(valid_messages) < 2:
            continue
            
        conversation_text = "\n".join(valid_messages)
        status = "Thành công" if phone_number else "Thất bại"
        
        processed.append({
            "STT": stt,
            "Nhãn thời gian": timestamp,
            "Tên Facebook": customer_name,
            "Facebook ID": customer_id,
            "Số điện thoại": phone_number,
            "Trạng thái": status,
            "Cuộc trò chuyện": conversation_text
        })
        stt += 1
        
    return processed

def export_to_csv(data: list, file_path: str):
    if not data:
        print("No data to export.")
        return
        
    df = pd.DataFrame(data)
    cols = ["STT", "Nhãn thời gian", "Tên Facebook", "Facebook ID", "Số điện thoại", "Trạng thái", "Cuộc trò chuyện"]
    if all(col in df.columns for col in cols):
        df = df[cols]
        
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

