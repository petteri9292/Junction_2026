import json
import torch
from huggingface_hub import model_info
from transformers import AutoTokenizer, AutoModel
from wikilens.database import connect
from wikilens.evidence import pack_vector
from pathlib import Path

model_id = "BAAI/bge-m3"
revision = "5617a9f61b028005a4858fdac845db406aefb181"
print("Model revision:", revision, flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = (
    AutoModel.from_pretrained(
        model_id, revision=revision, dtype=torch.float16, use_safetensors=False
    )
    .to("cuda")
    .eval()
)
query = "Winter War"
inputs = tokenizer([query], padding=True, truncation=False, return_tensors="pt").to(
    "cuda"
)
with torch.inference_mode():
    vector = (
        torch.nn.functional.normalize(
            model(**inputs).last_hidden_state[:, 0].float(), p=2, dim=1
        )[0]
        .cpu()
        .numpy()
    )
print("Query:", query, "Shape:", vector.shape, flush=True)
with connect() as db, db.cursor() as cursor:
    cursor.execute(
        """SELECT a.title,a.lang,a.url,a.text,n.distance
        FROM (SELECT lang,page_id,VEC_DISTANCE_COSINE(embedding,%s) AS distance
              FROM article_vector WHERE lang=%s ORDER BY distance LIMIT 10) n
        JOIN article a USING (lang,page_id) ORDER BY n.distance,a.page_id""",
        (pack_vector(vector), "fi"),
    )
    rows = cursor.fetchall()
result = {
    "query": query,
    "model": model_id,
    "revision": revision,
    "language": "fi",
    "results": rows,
}
Path("reports").mkdir(exist_ok=True)
Path("reports/winter-war-top10.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2)
)
for row in rows:
    row["text"] = " ".join(row["text"].split()[:20])
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
