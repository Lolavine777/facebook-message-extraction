# KẾ HOẠCH & QUY ƯỚC DỰ ÁN CRAWL DATA META GRAPH API

## 1. Technology Stack
- **Ngôn ngữ:** Python 3.10+
- **Thư viện chính:**
  - `requests`: Gọi API HTTP.
  - `pandas`: Xử lý, định dạng và xuất dữ liệu CSV.
  - `python-dotenv`: Load biến môi trường.
  - `tenacity`: Retry request khi bị rate limit (429).
- **Thư viện Testing:**
  - `pytest`: Framework test cốt lõi.
  - `responses`: Mock thư viện requests cho API Graph.
  - `pytest-mock`: Hỗ trợ mock.

## 2. Naming Conventions
- **Biến và Hàm:** `snake_case`
- **Lớp (Class):** `PascalCase`
- **Hằng số:** `UPPER_SNAKE_CASE` (ví dụ `PAGE_ID`, `MAX_RETRIES`)
- **Tên file:** `snake_case.py` (ví dụ `pipeline.py`, `main.py`)
- **Test files:** Bắt đầu bằng `test_` (ví dụ `test_pipeline.py`, `test_e2e.py`)

## 3. Hard Rules (Non-negotiable)
- **Luôn TDD (Test-Driven Development):** Không viết dòng code logic nào trước khi test thất bại (RED -> GREEN -> REFACTOR).
- **Bảo toàn PII phục vụ CRM:** Bóc tách dữ liệu SĐT và Facebook ID thật để phục vụ chăm sóc khách hàng nội bộ.
- **E2E Validation:** Tất cả 4 kịch bản E2E phải pass xanh.
- **Không Hardcode Token:** Token, Page ID và Version phải đưa vào `.env`.

## 4. Workflow
1. Đọc yêu cầu từ `task.md`.
2. Tạo file Test mô tả hành vi đúng, chạy cho báo lỗi.
3. Tạo file Code với logic tối thiểu để qua Test.
4. Refactor và chuyển sang task mới.
5. Cập nhật `task.md` liên tục.
