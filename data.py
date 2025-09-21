from datasets import load_dataset
import pickle, os

os.makedirs("datasets", exist_ok=True)

# Example: DailyDialog
dailydialog = load_dataset("daily_dialog", trust_remote_code=True)

# Save as pickle
import pickle
with open("datasets/dailydialog.pkl", "wb") as f:
    pickle.dump(dailydialog, f)

print("✅ DailyDialog saved in /datasets folder")
