import sys
import pipeline

def run_pipeline(output_file="chat_logs_dataset.csv"):
    try:
        print("Starting Meta Graph API Data Pipeline...")
        
        # 1. Load configuration
        config = pipeline.load_config()
        print(f"[*] Loaded config for Page ID: {config.get('PAGE_ID')}")
        
        # 2. Fetch conversations
        print("[*] Fetching conversations from Meta Graph API...")
        raw_data = pipeline.fetch_conversations(config)
        print(f"[*] Fetched {len(raw_data)} conversation nodes.")
        
        # 3. Process and clean
        print("[*] Processing and cleaning data...")
        processed_data = pipeline.process_conversations(raw_data, config)
        print(f"[*] Successfully processed {len(processed_data)} valid conversations.")
        
        # 4. Export
        print(f"[*] Exporting dataset to {output_file}...")
        pipeline.export_to_csv(processed_data, output_file)
        print("[+] Pipeline finished successfully!")
        
    except Exception as e:
        print(f"[-] Pipeline failed with error: {str(e)}", file=sys.stderr)
        raise e

if __name__ == "__main__":
    run_pipeline()
