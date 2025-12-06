
def write_formatted_report():
    report_text = """
Classification Report (Percentage)
================================

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Bear        | 43.0%     | 44.0%  | 43.0%    | 447     |
| Bull        | 54.0%     | 30.0%  | 38.0%    | 540     |
| Mid         | 12.0%     | 38.0%  | 18.0%    | 108     |
|             |           |        |          |         |
| Accuracy    |           |        | 36.0%    | 1095    |
| Macro Avg   | 36.0%     | 37.0%  | 33.0%    | 1095    |
| Weighted Avg| 45.0%     | 36.0%  | 38.0%    | 1095    |
"""
    print(report_text)
    with open("classification_report.txt", "w") as f:
        f.write(report_text.strip())
    print("Saved to classification_report.txt")

if __name__ == "__main__":
    write_formatted_report()
