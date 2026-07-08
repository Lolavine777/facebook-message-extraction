import pandas as pd
import re

def extract_better_phone(text):
    if not isinstance(text, str):
        return ""
    
    # Remove all spaces, dots, dashes before checking to make it easier, 
    # but we only want to extract phone-like sequences.
    # Better regex: optional + or nothing, then 84 or 0, then spaces/dots/dashes, then digits
    # Let's match sequences of digits, spaces, dots, dashes that have length between 10 and 15
    # A robust approach: find all sequences of digits in the string, join them, and check if it's a valid VN phone.
    # Actually, we can just use a regex that allows separators.
    pattern = r'(?:\+?84|0)(?:[\s\.\-]*\d){9}\b'
    match = re.search(pattern, text)
    if match:
        # Clean up the matched string to return just the digits
        raw_phone = match.group(0)
        clean_phone = re.sub(r'\D', '', raw_phone)
        # Normalize to start with 0 instead of 84 for consistency (optional)
        if clean_phone.startswith('84'):
            clean_phone = '0' + clean_phone[2:]
        return clean_phone
    return ""

def main():
    # Read CSV treating all columns as string to prevent loss of leading zeros and scientific notation for IDs
    df = pd.read_csv('chat_logs_dataset.csv', dtype=str)
    
    # Calculate initial count
    initial_count = df['Số điện thoại'].notna().sum() - (df['Số điện thoại'].isna() | (df['Số điện thoại'] == 'nan')).sum()
    
    # Apply to rows where status is 'Thất bại' or phone is missing/nan
    mask = df['Số điện thoại'].isna() | (df['Số điện thoại'] == 'nan') | (df['Trạng thái'] == 'Thất bại')
    
    # Extract phone from conversation
    extracted = df.loc[mask, 'Cuộc trò chuyện'].apply(extract_better_phone)
    
    # Count how many new phones we found
    new_phones = extracted[extracted != ""]
    print(f"Initial phone count: {initial_count}")
    print(f"Newly found phones: {len(new_phones)}")
    
    # Update dataframe
    df['Số điện thoại'] = df['Số điện thoại'].astype(str)
    # Re-mask after casting, keep nan as string 'nan' or mask using initial missing indices
    # Actually, we already have `new_phones.index`
    df.loc[new_phones.index, 'Số điện thoại'] = new_phones
    df.loc[new_phones.index, 'Trạng thái'] = 'Thành công'
    
    # Optional: replace 'nan' back with actual NaN if needed, but saving to csv will handle it or we leave it.
    df['Số điện thoại'] = df['Số điện thoại'].replace('nan', '')
    
    # Calculate final
    final_count = (df['Số điện thoại'] != '').sum()
    print(f"Final phone count: {final_count}")
    
    # Save back
    df.to_csv('chat_logs_dataset_recovered.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()
