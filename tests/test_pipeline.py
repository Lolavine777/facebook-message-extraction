import os
import sys
import pytest
import responses
import requests
import pandas as pd

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import load_config, remove_sales_templates, fetch_conversations, process_conversations, export_to_csv

def test_load_config_success(monkeypatch):
    # Mock environment variables and prevent load_dotenv
    monkeypatch.setattr("pipeline.load_dotenv", lambda: None)
    monkeypatch.setenv("PAGE_ID", "12345")
    monkeypatch.setenv("PAGE_ACCESS_TOKEN", "fake_token")
    monkeypatch.setenv("GRAPH_API_VERSION", "v25.0")
    
    config = load_config()
    
    assert config["PAGE_ID"] == "12345"
    assert config["PAGE_ACCESS_TOKEN"] == "fake_token"
    assert config["GRAPH_API_VERSION"] == "v25.0"

def test_load_config_missing_variables(monkeypatch):
    # Ensure environment is clear and prevent load_dotenv from reading local .env file
    monkeypatch.setattr("pipeline.load_dotenv", lambda: None)
    monkeypatch.delenv("PAGE_ID", raising=False)
    monkeypatch.delenv("PAGE_ACCESS_TOKEN", raising=False)
    
    with pytest.raises(ValueError, match="Missing required environment variables: PAGE_ID, PAGE_ACCESS_TOKEN"):
        load_config()

def test_remove_sales_templates():
    text = "Chào bạn! Đây là thông tin cơ sở: CS1: Hà Nội, CS2: Hồ Chí Minh, CS3: Đà Nẵng."
    assert remove_sales_templates(text) == "Chào bạn! Đây là thông tin cơ sở: [THÔNG_TIN_CƠ_SỞ]"
    assert remove_sales_templates("Chào bạn!") == "Chào bạn!"
    assert remove_sales_templates("Gọi cho tôi số 0901234567 nhé") == "Gọi cho tôi số 0901234567 nhé"

@responses.activate
def test_fetch_conversations_happy_path():
    config = {"PAGE_ID": "123", "PAGE_ACCESS_TOKEN": "token", "GRAPH_API_VERSION": "v25.0"}
    url = f"https://graph.facebook.com/{config['GRAPH_API_VERSION']}/{config['PAGE_ID']}/conversations?fields=updated_time,messages{{message,from,created_time}}&access_token={config['PAGE_ACCESS_TOKEN']}"
    
    mock_data = {
        "data": [
            {"id": "conv1", "messages": {"data": [{"id": "m1", "message": "hello"}]}}
        ]
    }
    responses.add(responses.GET, url, json=mock_data, status=200)
    
    data = fetch_conversations(config)
    assert len(data) == 1
    assert data[0]["id"] == "conv1"

@responses.activate
def test_fetch_conversations_pagination():
    config = {"PAGE_ID": "123", "PAGE_ACCESS_TOKEN": "token", "GRAPH_API_VERSION": "v25.0"}
    url1 = f"https://graph.facebook.com/{config['GRAPH_API_VERSION']}/{config['PAGE_ID']}/conversations?fields=updated_time,messages{{message,from,created_time}}&access_token={config['PAGE_ACCESS_TOKEN']}"
    url2 = "https://graph.facebook.com/v25.0/123/conversations?after=cursor"
    
    mock_data_1 = {
        "data": [{"id": "conv1"}],
        "paging": {"next": url2}
    }
    mock_data_2 = {
        "data": [{"id": "conv2"}]
    }
    
    responses.add(responses.GET, url1, json=mock_data_1, status=200)
    responses.add(responses.GET, url2, json=mock_data_2, status=200)
    
    data = fetch_conversations(config)
    assert len(data) == 2
    assert data[0]["id"] == "conv1"
    assert data[1]["id"] == "conv2"

@responses.activate
def test_fetch_conversations_429_retry():
    config = {"PAGE_ID": "123", "PAGE_ACCESS_TOKEN": "token", "GRAPH_API_VERSION": "v25.0"}
    url = f"https://graph.facebook.com/{config['GRAPH_API_VERSION']}/{config['PAGE_ID']}/conversations?fields=updated_time,messages{{message,from,created_time}}&access_token={config['PAGE_ACCESS_TOKEN']}"
    
    mock_data = {"data": [{"id": "conv1"}]}
    
    # First call returns 429, second returns 200
    responses.add(responses.GET, url, json={"error": {"code": 4, "message": "Application request limit reached"}}, status=429)
    responses.add(responses.GET, url, json=mock_data, status=200)
    
    data = fetch_conversations(config)
    
    assert len(data) == 1
    assert data[0]["id"] == "conv1"
    assert len(responses.calls) == 2

def test_process_conversations():
    config = {"PAGE_ID": "page123"}
    raw_data = [
        # Conversation 1: Normal with Phone Number
        {
            "id": "c1",
            "updated_time": "2026-07-06T10:00:00+0000",
            "messages": {
                "data": [
                    {"id": "m3", "from": {"id": "page123", "name": "My Page"}, "message": "Ok bạn", "created_time": "2026-07-06T10:00:00+0000"},
                    {"id": "m2", "from": {"id": "user456", "name": "User"}, "message": ""}, # Empty message, should be filtered
                    {"id": "m1", "from": {"id": "user456", "name": "John Doe"}, "message": "Số mình là 0901234567 nhé", "created_time": "2026-07-06T09:00:00+0000"}
                ]
            }
        },
        # Conversation 2: Only 1 interaction, should be dropped
        {
            "id": "c2",
            "updated_time": "2026-07-06T08:00:00+0000",
            "messages": {
                "data": [
                    {"id": "m1", "from": {"id": "user789", "name": "User 2"}, "message": "Vẫy tay", "created_time": "2026-07-06T08:00:00+0000"}
                ]
            }
        },
        # Conversation 3: No phone number
        {
            "id": "c3",
            "updated_time": "2026-07-05T10:00:00+0000",
            "messages": {
                "data": [
                    {"id": "m2", "from": {"id": "page123"}, "message": "Dạ", "created_time": "2026-07-05T10:00:00+0000"},
                    {"id": "m1", "from": {"id": "user999", "name": "Jane Doe"}, "message": "Alo shop", "created_time": "2026-07-05T09:00:00+0000"}
                ]
            }
        }
    ]
    
    processed = process_conversations(raw_data, config)
    
    assert len(processed) == 2
    
    assert processed[0]["STT"] == 1
    assert processed[0]["Tên Facebook"] == "John Doe"
    assert processed[0]["Facebook ID"] == "user456"
    assert processed[0]["Số điện thoại"] == "0901234567"
    assert processed[0]["Trạng thái"] == "Thành công"
    assert "2026" in processed[0]["Nhãn thời gian"]
    expected_chat_1 = "Khách hàng: Số mình là 0901234567 nhé\nPage: Ok bạn"
    assert processed[0]["Cuộc trò chuyện"] == expected_chat_1
    
    assert processed[1]["STT"] == 2
    assert processed[1]["Tên Facebook"] == "Jane Doe"
    assert processed[1]["Facebook ID"] == "user999"
    assert processed[1]["Số điện thoại"] == ""
    assert processed[1]["Trạng thái"] == "Thất bại"
    expected_chat_3 = "Khách hàng: Alo shop\nPage: Dạ"
    assert processed[1]["Cuộc trò chuyện"] == expected_chat_3

def test_export_to_csv(tmp_path):
    data = [
        {"STT": 1, "Nhãn thời gian": "2026-07-06", "Tên Facebook": "A", "Facebook ID": "1", "Số điện thoại": "0901234567", "Trạng thái": "Thành công", "Cuộc trò chuyện": "Khách hàng: Xin chào\nPage: Dạ chào anh"},
        {"STT": 2, "Nhãn thời gian": "2026-07-06", "Tên Facebook": "B", "Facebook ID": "2", "Số điện thoại": "", "Trạng thái": "Thất bại", "Cuộc trò chuyện": "Page: Chào bạn, có thể giúp gì không?"}
    ]
    
    file_path = tmp_path / "test_output.csv"
    export_to_csv(data, str(file_path))
    
    assert file_path.exists()
    
    df = pd.read_csv(file_path, encoding='utf-8-sig', dtype={'Số điện thoại': str}).fillna("")
    assert len(df) == 2
    assert list(df.columns) == ["STT", "Nhãn thời gian", "Tên Facebook", "Facebook ID", "Số điện thoại", "Trạng thái", "Cuộc trò chuyện"]
    assert df.iloc[0]["STT"] == 1
    assert df.iloc[0]["Số điện thoại"] == "0901234567"
    assert df.iloc[1]["Cuộc trò chuyện"] == "Page: Chào bạn, có thể giúp gì không?"
