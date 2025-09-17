import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os

# Initialize Firestore
cred = credentials.Certificate("firebase_key.json")
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

collection_name = "hydraulic_cycles"
collection_ref = db.collection(collection_name)

all_rows = []

docs = list(collection_ref.stream())
print(f"\nDocuments in '{collection_name}': {len(docs)}")

for doc in docs:
    doc_id = doc.id
    doc_dict = doc.to_dict()
    
    cycle_data = doc_dict.get("cycle_data")
    
    if not cycle_data:
        print(f"⚠️ No 'cycle_data' found for document {doc_id}")
        continue
    
    # If cycle_data is a list of dicts
    if isinstance(cycle_data, list):
        for row in cycle_data:
            row["cycle_id"] = doc_id
            all_rows.append(row)
    # If cycle_data is a single dict
    elif isinstance(cycle_data, dict):
        row = cycle_data
        row["cycle_id"] = doc_id
        all_rows.append(row)
    else:
        print(f"⚠️ Unsupported type for 'cycle_data' in {doc_id}")

# Export to CSV
if all_rows:
    df = pd.DataFrame(all_rows)
    output_file = os.path.join(os.getcwd(), "cycle_data_only.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Exported {len(df)} rows to '{output_file}'")
else:
    print("⚠️ No cycle_data found to export")
