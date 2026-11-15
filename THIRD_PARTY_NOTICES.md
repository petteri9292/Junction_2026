# Third party notices

## MariaDB Wikipedia RAG starter

The vector schema and indexed SQL query shape were informed by
https://github.com/MariaDB/mariadb-vector-wikipedia-rag.

MIT License

Copyright (c) 2026 MariaDB Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Wikipedia content

Wikipedia article text and the released derived vectors are CC BY-SA 4.0,
with attribution to Wikipedia and its authors. Source revision links appear in
the demo. FineWiki and the MariaDB Foundation prepared the challenge files.
The application MIT licence does not replace the content licence.

https://creativecommons.org/licenses/by-sa/4.0/


## Pretrained reranker comparison models

Model weights remain outside Git and are downloaded from their publishers.

- Jarbas/m2v-256-bge-reranker-large: Model2Vec static embeddings distilled by
  the publisher; model card declares MIT.
  https://huggingface.co/Jarbas/m2v-256-bge-reranker-large
- BAAI/bge-reranker-v2-m3: multilingual cross-encoder; model card declares
  Apache-2.0. https://huggingface.co/BAAI/bge-reranker-v2-m3
- Model2Vec library by Minish Lab: MIT. https://github.com/MinishLab/model2vec

This project uses pretrained inference only; it does not train or distill these
models. Recorded revisions are preserved in each comparison report.

## Live generation and embedding checkpoints

The live pipeline also downloads these pinned checkpoints from their publishers:

- Qwen/Qwen3-4B-Instruct-2507-FP8:
  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8
- BAAI/bge-m3: https://huggingface.co/BAAI/bge-m3

Weights are not bundled with this repository. Consult each publisher's model
card and licence for the terms that apply to the checkpoint. Exact revisions
used in the measurements are recorded in `docs/full-l-embeddings.md`.
