import os
import sys
import pytest
import responses
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("PAGE_ID", "123")
    monkeypatch.setenv("PAGE_ACCESS_TOKEN", "token")
    monkeypatch.setenv("GRAPH_API_VERSION", "v25.0")
    
@responses.activate
def test_e2e_scenario_1_happy_path(mock_env, tmp_path):
    # E2E Scenario 1: Cấu hình chuẩn (Happy Path)
    url = "https://graph.facebook.com/v25.0/123/conversations?fields=updated_time,messages{message,from,created_time}&access_token=token"
    mock_data = {
        "data": [
            {
                "id": "c1",
                "messages": {
                    "data": [
                        {"id": "m2", "from": {"id": "123"}, "message": "Chào bạn"},
                        {"id": "m1", "from": {"id": "456"}, "message": "Alo shop"}
                    ]
                }
            },
            {
                "id": "c2",
                "messages": {
                    "data": [
                        {"id": "m2", "from": {"id": "123"}, "message": "Dạ"},
                        {"id": "m1", "from": {"id": "789"}, "message": "Bao nhiêu tiền"}
                    ]
                }
            }
        ]
    }
    responses.add(responses.GET, url, json=mock_data, status=200)
    
    # Run pipeline
    output_file = str(tmp_path / "chat_logs_dataset.csv")
    main.run_pipeline(output_file=output_file)
    
    # Assertions
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file, encoding='utf-8-sig')
    assert len(df) == 2
    assert list(df.columns) == ["STT", "Nhãn thời gian", "Tên Facebook", "Facebook ID", "Số điện thoại", "Trạng thái", "Cuộc trò chuyện"]
    assert df.iloc[0]["STT"] == 1
    assert df.iloc[0]["Cuộc trò chuyện"] == "Khách hàng: Alo shop\nPage: Chào bạn"
    assert df.iloc[1]["STT"] == 2
    assert df.iloc[1]["Cuộc trò chuyện"] == "Khách hàng: Bao nhiêu tiền\nPage: Dạ"

@responses.activate
def test_e2e_scenario_2_pagination(mock_env, tmp_path):
    # E2E Scenario 2: Phân trang và nối dữ liệu
    url1 = "https://graph.facebook.com/v25.0/123/conversations?fields=updated_time,messages{message,from,created_time}&access_token=token"
    url2 = "https://graph.facebook.com/v25.0/123/conversations?after=cursor"
    
    mock_data_1 = {
        "data": [
            {
                "id": "c1",
                "messages": {
                    "data": [
                        {"id": "m2", "from": {"id": "123"}, "message": "Rep 1"},
                        {"id": "m1", "from": {"id": "456"}, "message": "Khách 1"}
                    ]
                }
            }
        ],
        "paging": {"next": url2}
    }
    
    mock_data_2 = {
        "data": [
            {
                "id": "c2",
                "messages": {
                    "data": [
                        {"id": "m2", "from": {"id": "123"}, "message": "Rep 2"},
                        {"id": "m1", "from": {"id": "789"}, "message": "Khách 2"}
                    ]
                }
            }
        ]
    }
    
    responses.add(responses.GET, url1, json=mock_data_1, status=200)
    responses.add(responses.GET, url2, json=mock_data_2, status=200)
    
    output_file = str(tmp_path / "chat_logs_dataset.csv")
    main.run_pipeline(output_file=output_file)
    
    df = pd.read_csv(output_file, encoding='utf-8-sig')
    assert len(df) == 2
    assert df.iloc[0]["STT"] == 1
    assert df.iloc[1]["STT"] == 2

@responses.activate
def test_e2e_scenario_3_cleansing_pii(mock_env, tmp_path):
    # E2E Scenario 3: Làm sạch dữ liệu và Ẩn danh
    url = "https://graph.facebook.com/v25.0/123/conversations?fields=updated_time,messages{message,from,created_time}&access_token=token"
    mock_data = {
        "data": [
            {
                "id": "c1", # PII and empty message
                "messages": {
                    "data": [
                        {"id": "m3", "from": {"id": "456"}, "message": "Sđt của bạn là 0901234567"},
                        {"id": "m2", "from": {"id": "456"}, "message": ""},
                        {"id": "m1", "from": {"id": "456"}, "message": "Chào page"}
                    ]
                }
            },
            {
                "id": "c2", # Only 1 valid message -> dropped
                "messages": {
                    "data": [
                        {"id": "m1", "from": {"id": "789"}, "message": "Vẫy tay"}
                    ]
                }
            }
        ]
    }
    responses.add(responses.GET, url, json=mock_data, status=200)
    
    output_file = str(tmp_path / "chat_logs_dataset.csv")
    main.run_pipeline(output_file=output_file)
    
    df = pd.read_csv(output_file, encoding='utf-8-sig', dtype={'Số điện thoại': str})
    assert len(df) == 1
    assert df.iloc[0]["Số điện thoại"] == "0901234567"
    assert df.iloc[0]["Trạng thái"] == "Thành công"

@responses.activate
def test_e2e_scenario_4_retry_429(mock_env, tmp_path):
    # E2E Scenario 4: Phục hồi sau lỗi (Rate Limiting)
    url = "https://graph.facebook.com/v25.0/123/conversations?fields=updated_time,messages{message,from,created_time}&access_token=token"
    mock_data = {
        "data": [
            {
                "id": "c1",
                "messages": {
                    "data": [
                        {"id": "m2", "from": {"id": "123"}, "message": "Dạ"},
                        {"id": "m1", "from": {"id": "456"}, "message": "Alo"}
                    ]
                }
            }
        ]
    }
    
    # 1st call fails, 2nd succeeds
    responses.add(responses.GET, url, json={"error": "Rate limit"}, status=429)
    responses.add(responses.GET, url, json=mock_data, status=200)
    
    output_file = str(tmp_path / "chat_logs_dataset.csv")
    main.run_pipeline(output_file=output_file)
    
    df = pd.read_csv(output_file, encoding='utf-8-sig')
    assert len(df) == 1
    assert len(responses.calls) == 2

@responses.activate
def test_e2e_scenario_5_server_error(mock_env, tmp_path):
    # E2E Scenario 5: Handle 500 Server Error gracefully
    url1 = "https://graph.facebook.com/v25.0/123/conversations?fields=updated_time,messages{message,from,created_time}&access_token=token"
    url2 = "https://graph.facebook.com/v25.0/123/conversations?after=cursor"
    
    mock_data_1 = {
        "data": [
            {
                "id": "c1",
                "messages": {
                    "data": [
                        {"id": "m2", "from": {"id": "123"}, "message": "Rep 1"},
                        {"id": "m1", "from": {"id": "456"}, "message": "Khách 1"}
                    ]
                }
            }
        ],
        "paging": {"next": url2}
    }
    
    responses.add(responses.GET, url1, json=mock_data_1, status=200)
    # url2 fails consistently with 500
    responses.add(responses.GET, url2, body="Internal Server Error", status=500)
    
    output_file = str(tmp_path / "chat_logs_dataset.csv")
    # Pipeline should NOT crash, but save data from url1
    main.run_pipeline(output_file=output_file)
    
    df = pd.read_csv(output_file, encoding='utf-8-sig')
    assert len(df) == 1
    assert df.iloc[0]["STT"] == 1
