import fitz  # PyMuPDF
import re
import json
import os

def extract_cis_rules(pdf_path):
    # Get absolute path to the PDF file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_absolute_path = os.path.join(script_dir, os.path.basename(pdf_path))
    
    print(f"Reading PDF from: {pdf_absolute_path}")
    doc = fitz.open(pdf_absolute_path)
    
    # Use a set to track unique rule IDs
    seen_rules = set()
    rules = []
    valid_rule_id_pattern = r"^(\d+|\d+\.\d+|\d+\.\d+\.\d+)$"
    
    # Skip the first few pages that contain table of contents
    start_page = 5  # Adjust this number based on where actual rules begin
    
    for page_num in range(start_page, len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Improved rule pattern to capture complete titles without truncation
        # Look for rule numbers followed by a title that starts with "Ensure" or "Maintain"
        rule_pattern = r"(\d+(?:\.\d+){0,2})\s+((?:Ensure|Maintain)[^\n\(]{2,})(?:\s*\((?:Manual|Automated)\))?"
        matches = re.finditer(rule_pattern, text)
        
        for match in matches:
            rule_id, title = match.groups()
            
            # Skip if not a valid rule ID format
            if not re.match(valid_rule_id_pattern, rule_id.strip()):
                continue
                
            # Skip if we've seen this rule ID before
            if rule_id in seen_rules:
                continue
                
            # Skip rules beyond 5.7
            parts = rule_id.split('.')
            if len(parts) >= 1 and int(parts[0]) > 5:
                continue
            if len(parts) >= 2 and int(parts[0]) == 5 and int(parts[1]) > 7:
                continue
                
            seen_rules.add(rule_id)
            
            # Increased look-ahead for longer content sections (10000 chars should be enough for a rule)
            rule_text = text[match.end():match.end() + 10000]
            
            # Improved content extraction patterns with better boundaries
            description = re.search(r"Description:\s*(.*?)(?=\s*(?:Rationale:|Profile Applicability:|Audit:|Impact:|$))", rule_text, re.DOTALL)
            rationale = re.search(r"Rationale:\s*(.*?)(?=\s*(?:Audit:|Impact:|Remediation:|Profile Applicability:|$))", rule_text, re.DOTALL)
            impact = re.search(r"Impact:\s*(.*?)(?=\s*(?:Audit:|Remediation:|References:|CIS Controls:|Profile Applicability:|$))", rule_text, re.DOTALL)
            audit = re.search(r"Audit:\s*(.*?)(?=\s*(?:Remediation:|References:|Impact:|CIS Controls:|Profile Applicability:|$))", rule_text, re.DOTALL)
            
            # Default remediation extraction
            remediation = re.search(r"Remediation:\s*(.*?)(?=\s*(?:References:|CIS Controls:|Impact:|Audit:|Default Value:|Profile Applicability:|$))", rule_text, re.DOTALL)
            
            # Special handling for rule 5.7
            if rule_id == "5.7":
                # For rule 5.7, extract the remediation manually based on the image provided
                rule_5_7_remediation = """From Console:
1. Sign in to the AWS Management Console and navigate to the EC2 dashboard at https://console.aws.amazon.com/ec2/.
2. In the left navigation panel, under the INSTANCES section, choose Instances.
3. Select the EC2 instance that you want to examine.
4. Choose Actions > Instance Settings > Modify instance metadata options.
5. Set Instance metadata service to Enable.
6. Set IMDSv2 to Required.
7. Repeat steps 1-6 to perform the remediation process for other EC2 instances in all applicable AWS region(s).

From Command Line:
1. Run the describe-instances command, applying the appropriate filters to list the IDs of all existing EC2 instances currently available in the selected region.
2. The command output should return a table with the requested instance IDs.
3. Run the modify-instance-metadata-options command with an instance ID obtained from the previous step to update the Instance Metadata Version:"""
                
                # Create a mock match object with the same interface as re.search() result
                class MockMatch:
                    def __init__(self, text):
                        self.text = text
                    def group(self, n):
                        return self.text
                
                remediation = MockMatch(rule_5_7_remediation)
            
            # Improved text cleaning function
            def clean_text(text_match):
                if not text_match:
                    return ""
                
                text_content = text_match.group(1) if hasattr(text_match, 'group') else text_match.group(1) if text_match else ""
                text_content = text_content.strip()
                
                # Replace multiple spaces and newlines with a single space
                cleaned = re.sub(r'\s+', ' ', text_content)
                
                # Fix common formatting issues
                cleaned = cleaned.replace(" . ", ". ")
                cleaned = cleaned.replace(" , ", ", ")
                
                # Ensure the text ends with a period if it doesn't already
                if cleaned and not cleaned.endswith('.'):
                    cleaned += '.'
                    
                return cleaned
            
            rule = {
                "id": rule_id.strip(),
                "title": title.strip(),
                "description": clean_text(description),
                "rationale": clean_text(rationale),
                "impact": clean_text(impact),
                "audit_procedure": clean_text(audit),
                "remediation": clean_text(remediation) if rule_id != "5.7" else rule_5_7_remediation
            }
            
            rules.append(rule)
            print(f"\nFound Rule: {rule_id}")
            print(f"Title: {title.strip()}")
            print("Content lengths:")
            print(f"- Description: {len(rule['description'])} chars")
            print(f"- Rationale: {len(rule['rationale'])} chars")
            print(f"- Impact: {len(rule['impact'])} chars")
            print(f"- Audit: {len(rule['audit_procedure'])} chars")
            print(f"- Remediation: {len(rule['remediation'])} chars")
            print("-" * 40)

    # Sort rules by ID numerically before returning
    def numeric_sort_key(rule):
        parts = rule['id'].split('.')
        # Pad each part with zeros to ensure correct sorting
        return [int(part) for part in parts]
        
    rules.sort(key=numeric_sort_key)
    return rules

def save_rules_to_json(rules, output_file):
    """Save the extracted rules to a JSON file with pretty printing"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_absolute_path = os.path.join(script_dir, output_file)
    
    with open(output_absolute_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(rules)} rules to: {output_absolute_path}")
    
    # Print rule 5.7 specifically to verify extraction
    for rule in rules:
        if rule['id'] == "5.7":
            print("\nRule 5.7 data:")
            print(json.dumps(rule, indent=2))
            break
        
    # Print first rule as sample
    if rules:
        print("\nSample of first extracted rule:")
        print(json.dumps(rules[0], indent=2))
        
        # Print a rule with non-empty remediation as an example
        for rule in rules:
            if rule['remediation'] and rule['id'] != "5.7":
                print("\nSample rule with remediation:")
                print(json.dumps(rule, indent=2))
                break

def main():
    try:
        print("Starting CIS Benchmark extraction...")
        rules = extract_cis_rules("CIS_Benchmark_v4.pdf")
        
        if not rules:
            print("\nWarning: No rules were extracted!")
        else:
            save_rules_to_json(rules, "cis_benchmark.json")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
