import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from wikilens.database import connect
from wikilens.evidence import pack_vector

start = time.perf_counter()
root = Path("data/s-2026-10")
revision = "5617a9f61b028005a4858fdac845db406aefb181"
tok = AutoTokenizer.from_pretrained(
    "BAAI/bge-m3", revision=revision, local_files_only=True
)
model = (
    AutoModel.from_pretrained(
        "BAAI/bge-m3",
        revision=revision,
        local_files_only=True,
        use_safetensors=False,
        dtype=torch.float16,
    )
    .to("cuda")
    .eval()
)


def embeddings(text):
    inputs = tok([text], return_tensors="pt", truncation=False).to("cuda")
    with torch.inference_mode():
        h = model(**inputs).last_hidden_state.float()
        mask = inputs["attention_mask"].unsqueeze(-1)
        cls = torch.nn.functional.normalize(h[:, 0], dim=1)[0].cpu().numpy()
        mean = (
            torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=1)[0]
            .cpu()
            .numpy()
        )
    return cls, mean


query, query_mean = embeddings("Hanoi Rocks")
report = {
    "query": "Hanoi Rocks",
    "revision": revision,
    "same_article_checks": [],
    "source_db_checks": [],
    "rankings": {},
}
for lang in ("fi", "en", "sv"):
    a = pd.read_parquet(root / f"articles-{lang}.parquet")
    v = pd.read_parquet(root / f"vectors-bge-m3-{lang}.parquet")
    merged = a.merge(v, on=["lang", "page_id"], validate="one_to_one")
    candidates = merged[
        (~merged.truncated) & merged.text.str.len().between(200, 900)
    ].head(3)
    for r in candidates.itertuples():
        text = f"{r.title}\n\n{r.text}"
        cls, mean = embeddings(text)
        stored = np.asarray(r.embedding, dtype=np.float32)
        unit = stored / np.linalg.norm(stored)
        report["same_article_checks"].append(
            {
                "lang": lang,
                "page_id": int(r.page_id),
                "title": r.title,
                "tokens": len(tok(text)["input_ids"]),
                "stored_norm": float(np.linalg.norm(stored)),
                "cls_cosine": float(unit @ cls),
                "mean_cosine": float(unit @ mean),
            }
        )
    matrix = np.stack(v.embedding).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    report.setdefault("vector_norms", {})[lang] = {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "nonfinite": int((~np.isfinite(matrix)).sum()),
    }
    if lang == "fi":
        distances = 1 - (matrix @ query) / norms
        ids = np.argsort(distances)[:10]
        titles = a.set_index("page_id")
        report["rankings"]["exact_numpy_cls"] = [
            {
                "page_id": int(v.iloc[i].page_id),
                "title": titles.loc[v.iloc[i].page_id, "title"],
                "distance": float(distances[i]),
            }
            for i in ids
        ]
        md = 1 - (matrix @ query_mean) / norms
        report["rankings"]["exact_numpy_mean"] = [
            {
                "page_id": int(v.iloc[i].page_id),
                "title": titles.loc[v.iloc[i].page_id, "title"],
                "distance": float(md[i]),
            }
            for i in np.argsort(md)[:5]
        ]
    with connect() as db, db.cursor() as c:
        for r in candidates.itertuples():
            c.execute(
                "SELECT embedding FROM article_vector WHERE lang=%s AND page_id=%s",
                (lang, int(r.page_id)),
            )
            dbv = np.frombuffer(c.fetchone()["embedding"], dtype="<f4")
            report["source_db_checks"].append(
                {
                    "lang": lang,
                    "page_id": int(r.page_id),
                    "exact_float32_match": bool(
                        np.array_equal(dbv, np.asarray(r.embedding, dtype=np.float32))
                    ),
                }
            )
with connect() as db, db.cursor() as c:
    for ef in (80, 400):
        c.execute("SET SESSION mhnsw_ef_search=%s", (ef,))
        c.execute(
            """SELECT a.title,a.page_id,n.distance FROM
        (SELECT lang,page_id,VEC_DISTANCE_COSINE(embedding,%s) AS distance
        FROM article_vector WHERE lang=%s ORDER BY distance LIMIT 10) n
        JOIN article a USING(lang,page_id) ORDER BY n.distance""",
            (pack_vector(query), "fi"),
        )
        report["rankings"][f"mariadb_ef_{ef}"] = c.fetchall()
report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
Path("reports/embedding-compatibility.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
