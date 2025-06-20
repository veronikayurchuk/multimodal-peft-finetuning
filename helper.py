import re
import json

def extract_json_from_llm_output(text) -> dict:
    """
    Extract JSON values for specified keys from text containing JSON-like content.
    
    Args:
        text (str): Text containing JSON-like content
        
    Returns:
        dict: Extracted values for 'menu', 'sub_total', and 'total' keys
    """
    # The keys we want to extract
    target_keys = ['menu', 'sub_total', 'total']
    result = {}
    
    # Find all content between single quotes that looks like JSON
    json_pattern = r"'(" + '|'.join(target_keys) + r")':\s*({[^}]+}|\[[^\]]+\])"
    matches = re.finditer(json_pattern, text)
    
    for match in matches:
        key = match.group(1)
        value_str = match.group(2)
        
        # Clean up the value string by replacing single quotes with double quotes
        value_str = value_str.replace("'", '"')
        
        try:
            # Try to parse the value as JSON
            value = json.loads(value_str)
            result[key] = value
        except json.JSONDecodeError:
            # If parsing fails, store the raw string
            result[key] = value_str
            
    return result

# Example usage
if __name__ == "__main__":
    sample_text = """your receipt text here... {'menu': {'nm': '1.0amel Black Tea  'priceprice': '28,000', 'cnt': '1X', 'price': '28,000'}, 'sub_total': {'subtotal_price': '28,000'}, 'total': {'total_price': '28,000', 'cashprice': '28,000'}}"""
    
    result = extract_json_values(sample_text)
    print(json.dumps(result, indent=2))
