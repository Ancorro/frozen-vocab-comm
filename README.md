# frozen-vocab-comm

Can two language models that were trained separately, but share one frozen vocabulary, read each other's hidden
states? "Embeddings Are Overpriced?" (Bathula, 2026) trains small models whose input and output vocabulary is a fixed
table of 512-bit codes, never trained. Two such models with the same table map tokens in and out of the same code
space. This project tests whether that shared space is enough for one model to pass its internal state to the other
directly, with no learned translator between them.

Who it is for: research on latent communication between LLM (large language model) agents.

Built with: Python, PyTorch, the paper's released table and checkpoints (downloaded at runtime, not committed).

## Method

The only space two frozen-vocabulary models share without training is the 512-dimensional code space: a token
enters as its code through a learned map `inp` (512 to 640), and the model's last hidden state leaves through `out`
(640 to 512) to be scored against every code. So the zero-translator channel is: the sender X's output vector
`out(h_X)` goes straight into the receiver Y's `inp`, in place of a token's code.

Each test sequence is a 128-token prefix, a 32-token span and 64 tokens after it, from the paper's held-out
validation tokens. Y always sees the true prefix and the true tokens after the span. The span is replaced by a
message, and the score is Y's mean next-token loss (nats) on the 64 tokens after the span. Lower means the message
carried more. For position t of the span, X sees the true text up to t-1 and its message stands for its guess of
token t.

| message | what Y receives at each span position |
| --- | --- |
| true | the real token's code (upper reference) |
| soft | X's raw output vector, rescaled to a code's length (the zero-translator channel) |
| sign | the sign of X's output vector: 512 bits, not always any token's code |
| mix | X's next-token probabilities averaged over the codes, a "soft token" |
| argmax | the code of X's top token: ordinary text, the baseline to beat |
| random | random ±1 codes (floor) |
| drop | span removed, prefix joined to the rest (floor) |

The paper's released B (float body) and C (ternary body) are not independent models: `train.py` seeds both bodies
with seed 0 and both saw the same batches in the same order. They differ in body precision only. Pairs B→C and C→B
therefore test a shared vocabulary *plus* a shared start; B→B and C→C (a model sending to itself) are the
easiest case the channel could face.

## Results

Hyperparameters: released checkpoints `B/step_07820.pt` and `C/step_07820.pt` (2.05B FineWeb-Edu tokens each,
Gemma-3-270M body, 18 layers, width 640); 256 sequences drawn from `data/val.npy` with seed 0; prefix 128, span 32,
after-span 64 tokens; bfloat16 autocast; batch 8; one RTX 3090; about 3 minutes in all. Raw per-sequence values
are in `results/channel.json`.

**Check of the rebuilt models.** Validation loss on 64 sequences of 1,024 tokens: B 3.373, C 3.451. The paper
reports 3.358 and 3.439 on 512 sequences, so the rebuild loads the checkpoints correctly.

**Receiver loss after the span** (nats per token, lower is better):

| sender → receiver | true | soft | sign | mix | argmax | random | drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B → C | 3.465 | 3.700 | 3.666 | 3.659 | **3.650** | 3.769 | 3.717 |
| C → B | 3.375 | 3.626 | 3.593 | 3.601 | **3.565** | 3.686 | 3.638 |
| B → B | 3.375 | 3.621 | 3.590 | 3.596 | **3.566** | 3.686 | 3.638 |
| C → C | 3.465 | 3.710 | 3.675 | 3.663 | **3.650** | 3.769 | 3.717 |

**Differences per sequence, with 95% bootstrap intervals** (positive: the message is worse than argmax):

| pair | soft − argmax | sign − argmax | mix − argmax | soft − drop |
| --- | --- | --- | --- | --- |
| B → C | +0.049 [+0.037, +0.062] | +0.016 [+0.004, +0.028] | +0.009 [+0.001, +0.017] | −0.017 [−0.031, −0.005] |
| C → B | +0.061 [+0.047, +0.076] | +0.028 [+0.016, +0.040] | +0.036 [+0.026, +0.046] | −0.012 [−0.027, +0.004] |
| B → B | +0.055 [+0.040, +0.070] | +0.024 [+0.011, +0.038] | +0.030 [+0.020, +0.041] | −0.017 [−0.033, −0.000] |
| C → C | +0.060 [+0.047, +0.074] | +0.025 [+0.013, +0.037] | +0.013 [+0.005, +0.022] | −0.007 [−0.020, +0.007] |

**Finding: the shared frozen vocabulary is not a usable hidden-state channel for these models.** In every pair,
the sender's top token beats every soft message. The raw output vector, the zero-translator channel this project
set out to test, is the worst message: it recovers about a quarter of what the top token recovers over dropping the
span, and in two pairs it is not clearly better than sending nothing. The cross-model result looks the same as a
model sending to itself (B→C matches C→C, C→B matches B→B), so the failure is not about crossing between models.
These models simply never learned to read anything at `inp` except exact token codes. Their output vectors live
in a different region of the code space from the ±1 token codes they were trained to read.

Limits: one pair of models that share their initialization; one span length and one position; no training of
any kind. A receiver trained on mixed soft inputs, or two models trained from different seeds, could behave
differently.

## How to run

```
bash setup.sh                       # PyPI install; CPU is enough for the tests
python -m pytest                    # tiny random models, under a minute on CPU
python scripts/download.py          # paper's B and C checkpoints and validation tokens into data/ (about 3.5 GB)
python scripts/check_val.py 64      # rebuilt models against the paper's validation loss (GPU)
python scripts/run_channel.py --n 256 --batch 8   # the channel test (GPU, about 3 minutes on an RTX 3090)
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
