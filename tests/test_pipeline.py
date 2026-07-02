import os
import sys
import pytest
import responses
import requests
import pandas as pd

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import load_config, clean_and_mask_pii, fetch_conversations, process_conversations, export_to_csv

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

def test_clean_and_mask_pii():
    assert clean_and_mask_pii("Gọi cho tôi số 0901234567 nhé") == "Gọi cho tôi số [SĐT] nhé"
    assert clean_and_mask_pii("Hoặc số 84901234567") == "Hoặc số [SĐT]"
    assert clean_and_mask_pii("Email là test@example.com nha") == "Email là [EMAIL] nha"
    assert clean_and_mask_pii("Liên hệ admin@domain.vn, sdt 0123456789.") == "Liên hệ [EMAIL], sdt [SĐT]."
    assert clean_and_mask_pii("Chào bạn!") == "Chào bạn!"

@responses.activate
def test_fetch_conversations_happy_path():
    config = {"PAGE_ID": "123", "PAGE_ACCESS_TOKEN": "token", "GRAPH_API_VERSION": "v25.0"}
    url = f"https://graph.facebook.com/{config['GRAPH_API_VERSION']}/{config['PAGE_ID']}/conversations?fields=messages{{message,from,created_time}}&access_token={config['PAGE_ACCESS_TOKEN']}"
    
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
    url1 = f"https://graph.facebook.com/{config['GRAPH_API_VERSION']}/{config['PAGE_ID']}/conversations?fields=messages{{message,from,created_time}}&access_token={config['PAGE_ACCESS_TOKEN']}"
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
    url = f"https://graph.facebook.com/{config['GRAPH_API_VERSION']}/{config['PAGE_ID']}/conversations?fields=messages{{message,from,created_time}}&access_token={config['PAGE_ACCESS_TOKEN']}"
    
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
        # Conversation 1: Normal with PII and empty message
        {
            "id": "c1",
            "messages": {
                "data": [
                    {"id": "m3", "from": {"id": "page123", "name": "My Page"}, "message": "Số 0901234567 nhé"},
                    {"id": "m2", "from": {"id": "user456", "name": "User"}, "message": ""}, # Empty message, should be filtered
                    {"id": "m1", "from": {"id": "user456", "name": "User"}, "message": "Chào bạn"}
                ]
            }
        },
        # Conversation 2: Only 1 interaction, should be dropped
        {
            "id": "c2",
            "messages": {
                "data": [
                    {"id": "m1", "from": {"id": "user789", "name": "User"}, "message": "Vẫy tay"}
                ]
            }
        },
        # Conversation 3: Just testing correct order (reverse from API response)
        {
            "id": "c3",
            "messages": {
                "data": [
                    {"id": "m2", "from": {"id": "page123"}, "message": "Dạ"},
                    {"id": "m1", "from": {"id": "user999"}, "message": "Alo"}
                ]
            }
        }
    ]
    
    processed = process_conversations(raw_data, config)
    
    assert len(processed) == 2
    
    assert processed[0]["STT"] == 1
    expected_chat_1 = "Khách hàng: Chào bạn\nPage: Số [SĐT] nhé"
    assert processed[0]["Cuộc trò chuyện"] == expected_chat_1
    
    
    assert processed[1]["STT"] == 2
    expected_chat_3 = "Khách hàng: Alo\nPage: Dạ"
    assert processed[1]["Cuộc trò chuyện"] == expected_chat_3

def test_export_to_csv(tmp_path):
    data = [
        {"STT": 1, "Cuộc trò chuyện": "Khách hàng: Xin chào\nPage: Dạ chào anh"},
        {"STT": 2, "Cuộc trò chuyện": "Page: Chào bạn, có thể giúp gì không?"}
    ]
    
    file_path = tmp_path / "test_output.csv"
    export_to_csv(data, str(file_path))
    
    assert file_path.exists()
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    assert len(df) == 2
    assert list(df.columns) == ["STT", "Cuộc trò chuyện"]
    assert df.iloc[0]["STT"] == 1
    assert df.iloc[1]["Cuộc trò chuyện"] == "Page: Chào bạn, có thể giúp gì không?"
