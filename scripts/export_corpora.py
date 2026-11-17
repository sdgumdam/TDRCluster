"""从 intake(case)+large_intent_label 聚合导出双路语料 (case 级, 199k)."""
import csv, os
from pymongo import MongoClient

uri = os.environ["MONGO_URI"]
client = MongoClient(uri, serverSelectionTimeoutMS=8000)
db = client["ai_agent"]

pipeline = [
    {"$sort": {"case_id": 1, "attempt": 1, "created_at": 1}},
    {"$group": {"_id": "$case_id", "intent": {"$last": "$intent"}, "caps": {"$last": "$capabilities"}}},
    {"$lookup": {"from": "case", "localField": "_id", "foreignField": "case_id", "as": "c"}},
    {"$unwind": "$c"},
    {"$match": {"c.input_raw.prompt": {"$type": "string", "$ne": ""}}},
    {"$project": {"case_id": "$_id", "prompt": "$c.input_raw.prompt",
                  "intent": {"$ifNull": ["$intent", ""]},
                  "caps": {"$ifNull": ["$caps", []]}}},
]
n = 0
with open("data/raw_data/corpus_intent.csv", "w", newline="", encoding="utf-8") as fa, \
     open("data/raw_data/corpus_capability.csv", "w", newline="", encoding="utf-8") as fb:
    wa, wb = csv.writer(fa), csv.writer(fb)
    wa.writerow(["qid", "query"]); wb.writerow(["qid", "query"])
    for doc in db.large_intent_label.aggregate(pipeline, allowDiskUse=True):
        case_id, prompt = doc["case_id"], doc["prompt"]
        intent, caps = doc["intent"], [c for c in doc["caps"] if c]
        wa.writerow([case_id, f"原始prompt：{prompt}\n任务意图：{intent}"])
        wb.writerow([case_id, f"原始prompt：{prompt}\n能力需求：{'；'.join(caps)}"])
        n += 1
print("rows:", n)
