from pathlib import Path
import pandas as pd

def main():
    path = Path("../data/Phishing_validation_emails.csv")
    df = pd.read_csv(path)
    possible_text_cols = []

    for col in df.columns:
        lowered = col.lower()
        if "text" in lowered or "email" in lowered:
            possible_text_cols.append(col)

    picked_col = possible_text_cols[0] if possible_text_cols else df.columns[0]
    df["raw_text"] = df[picked_col]
    df["text"] = df[picked_col].astype(str).str.strip()

    total = len(df)
    unique_texts = df["text"].nunique()

    print("External dataset duplicate check")
    print(f"Total rows: {total}")
    print(f"Unique texts:           {unique_texts}")
    print("Top 5 most used templates")
    print(df["text"].value_counts().head(5))

if __name__ == "__main__":
    main()
