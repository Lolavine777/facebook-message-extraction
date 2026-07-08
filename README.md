# Meta Graph API - Chat Data Extraction Pipeline

Dự án này cung cấp một pipeline hoàn chỉnh để tự động crawl (trích xuất) tin nhắn và cuộc gọi từ Fanpage Facebook thông qua Meta Graph API, đồng thời xử lý, làm sạch và chuẩn bị dữ liệu (Data Preparation) để phục vụ cho các hệ thống CRM, Telesales hoặc huấn luyện mô hình AI (Chatbot, NLP).

## 🌟 Tính Năng Chính
1. **Tự động trích xuất:** Lấy hàng chục nghìn lịch sử trò chuyện trên Fanpage sử dụng `PAGE_ID` và `PAGE_ACCESS_TOKEN` thông qua Meta Graph API.
2. **Trích xuất thông tin khách hàng (PII):** Tự động bắt số điện thoại Việt Nam (kể cả các định dạng khó như có dấu chấm, gạch ngang, thiếu số 0) và định danh PSID (Page-Scoped ID) của khách hàng.
3. **Làm sạch văn bản:** Loại bỏ các đoạn tin nhắn mẫu tự động, kịch bản bán hàng (sales templates) để làm sạch nội dung hội thoại thực sự.
4. **Phân tích EDA & Clustering:** Tích hợp pipeline để phân cụm ý định (Intent Clustering) và xuất biểu đồ trực quan (Wordcloud, Bar charts).
5. **Chuẩn bị dữ liệu AI:** Export dữ liệu ra chuẩn `JSONL` tương thích ngay với các mô hình Fine-tuning của OpenAI hoặc Alpaca (LLaMA).

## 🚀 Hướng Dẫn Cài Đặt (Reproducibility)

### 1. Yêu cầu hệ thống
* Python 3.10+
* Git

### 2. Cài đặt thư viện
Clone repository này về máy và cài đặt môi trường:
```bash
git clone <your-repo-url>
cd <your-repo-folder>

# Tạo virtual environment (khuyến nghị)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Cài đặt các thư viện cần thiết cho việc Crawl dữ liệu
pip install -r requirements.txt

# Cài đặt các thư viện cần thiết cho việc Phân tích và Machine Learning (Tuỳ chọn)
pip install -r requirements_analysis.txt
```

### 3. Cấu hình bảo mật (Môi trường)
Dự án sử dụng `.env` để bảo vệ các thông tin nhạy cảm (Token, ID). File này đã được đưa vào `.gitignore` để tránh bị đẩy nhầm lên public.
1. Copy file `.env.example` thành `.env`:
   ```bash
   cp .env.example .env
   ```
2. Mở file `.env` và điền thông tin thật của Fanpage bạn:
   ```env
   PAGE_ID=1234567890
   PAGE_ACCESS_TOKEN=EAAGXXXXXX...
   GRAPH_API_VERSION=v25.0
   ```
*(Lưu ý: Không bao giờ commit file `.env` lên GitHub)*

## 💡 Cách Sử Dụng

**1. Khởi chạy Pipeline Cào dữ liệu:**
```bash
python main.py
```
*Kết quả:* Sẽ tạo ra file `chat_logs_dataset.csv` chứa lịch sử chat và số điện thoại.

**2. Phục hồi Số điện thoại bị miss (Nâng cao):**
Nếu dataset gặp tình trạng sót số điện thoại do khách viết sai định dạng, chạy script sau để lọc lại nội dung chat và bổ sung số điện thoại:
```bash
python recover_phones.py
```
*Kết quả:* Sẽ tạo ra file `chat_logs_dataset_recovered.csv`.

**3. Phân tích Dữ liệu và Chuẩn bị AI:**
```bash
python ai_data_prep.py
```
*Kết quả:* Tạo ra file `intent_clusters.csv` (phân cụm ý định) và các định dạng huấn luyện AI `train_openai.jsonl`, `train_alpaca.jsonl`.

## 🛡️ Cam Kết Về Quyền Riêng Tư (Privacy)
Repository này tuân thủ các quy tắc bảo mật:
* **Không chứa dữ liệu thật:** Toàn bộ file `.csv`, `.jsonl` sinh ra từ quá trình trích xuất đều đã được loại trừ (ignored) trong file `.gitignore`.
* **Không hardcode Token:** Mọi mã định danh đều sử dụng `dotenv` và truyền từ file `.env`.

## 🧪 Testing
Chạy toàn bộ các test case sử dụng `pytest` để đảm bảo code hoạt động chính xác (đã bao gồm các mock test cho Graph API):
```bash
pytest tests/
```
