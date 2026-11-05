# frozen-vocab-comm

Can two language models that were trained separately, but share one frozen vocabulary, read each other's hidden
states? "Embeddings Are Overpriced?" (Bathula, 2026) trains small models whose input and output vocabulary is a fixed
table of 512-bit codes, never trained. Two such models with the same table map tokens in and out of the same code
space. This project tests whether that shared space is enough for one model to pass its internal state to the other
directly, with no learned translator between them.

Who it is for: research on latent communication between LLM (large language model) agents.

Built with: Python, PyTorch, the paper's released table and checkpoints (downloaded at runtime, not committed).

## How to run

```
bash setup.sh
```

## Sources and attribution

Every paper, repository, dataset, model, and asset this project uses or builds on, with its authors, link, and license.
Code copied or adapted from elsewhere also names its source in a comment at the top of the file.

- Sai Kiran Bathula. *Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models.*
  Preprint, 2026. [arXiv 2610.22428](https://arxiv.org/abs/2610.22428), DOI
  [10.5281/zenodo.23382634](https://doi.org/10.5281/zenodo.23382634). The question here grows from this paper.
- Code, frozen table and checkpoints: [huggingface.co/QLNI/embeddings-are-overpriced](https://huggingface.co/QLNI/embeddings-are-overpriced),
  QLNI Non-Commercial Research Licence (non-commercial research and education only). This repo downloads them at
  runtime and does not redistribute them. The table and the B and C checkpoints derive from Gemma 3 270M's embedding
  and carry [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- FineWeb-Edu (Penedo et al., [arXiv 2406.17557](https://arxiv.org/abs/2406.17557)), ODC-By 1.0: the text the
  checkpoints were trained on.

## Repo rules

This repo can be made public at any time, history included.

- No secrets: keys go in `.env` (untracked); `.env.example` lists the names only.
- No private information: cluster paths, host names, run logs, costs and personal notes go in `private/`, which is untracked and backed up outside git.
- Large data stays out of git; a script downloads it.
- Sources: add each new source to "Sources and attribution" in the commit that starts using it.
- Keep the real history: no squashing or force-pushing published commits.
