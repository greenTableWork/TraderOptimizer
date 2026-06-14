# TraderOptimizer Regime Detection Paper Lookup

Date: 2026-05-31

Purpose: seed a strategy-regime research corpus for representing what a
strategy config was tuned for. Regime labels are market inputs to tuning, not
benchmark pass/fail statuses.

## Local Sources Checked

- `TraderLab/paper_sources.md` lists arXiv q-fin, SSRN, SSRN finance journal browse pages, and RePEc/IDEAS.
- `TraderLab/research/papers/ssrn-3947905.pdf` is already local and matches [Clustering Market Regimes Using the Wasserstein Distance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905).
- `TraderLab/research/papers/ScienceDirect_articles_22May2026_23-30-29/Time-series-momentum_2012_Journal-of-Financial-Economics.pdf` is already local. It is useful for momentum/regime strategy behavior, but it is not itself a regime detector.

## Highest Source H-Index Additions

These are the top two regime-relevant additions found today where a source
h-index was observable from public journal-ranking pages.

| Rank | Paper | Source h-index evidence | Why it matters for Trader |
| --- | --- | --- | --- |
| 1 | [Market regime detection via realized covariances](https://ideas.repec.org/a/eee/ecmode/v111y2022ics0264999322000785.html), Andrea Bucci and Vito Ciciretti, 2022 | Economic Modelling, SCIMAGO h-index 126 from Research.com | Labels calm versus volatile regimes from covariance information and evaluates a regime-switching investment strategy. Directly maps to volatility, market stress, and cross-asset/correlation tuning regions. |
| 2 | [Regime-Switching Factor Investing with Hidden Markov Models](https://www.mdpi.com/1911-8074/13/12/311), Matthew Wang, Yi-Hong Lin, and Ilya Mikhelson, 2020 | Journal of Risk and Financial Management, SCIMAGO h-index 54 from Research.com | Maps HMM states to factor-model selection. Useful for representing strategy-family routing under market regimes rather than treating up/down as benchmark status. |

Adjacent high-h-index local seed: [Time-Series Momentum](https://doi.org/10.1016/j.jfineco.2011.11.003) is in Journal of Financial Economics, whose SCIMAGO h-index is 331. Keep it in the corpus as a strategy behavior and momentum benchmark paper, not as a regime-detection source.

## Core Regime Papers To Download Or Inspect

| Priority | Paper | Source | Method | Trader regime mapping | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | [Regime Changes and Financial Markets](https://www.nber.org/papers/w17182), Andrew Ang and Allan Timmermann, 2011 working paper and 2012 Annual Review article | NBER / Annual Review of Financial Economics | Review of Markov regime-switching models in finance | Canonical framework for regime labels, regime persistence, cross-covariance changes, and portfolio-choice consequences | Download NBER PDF or inspect Annual Review |
| 2 | [How do Regimes Affect Asset Allocation?](https://www.nber.org/papers/w10080), Andrew Ang and Geert Bekaert, 2003 | NBER / Financial Analysts Journal version | Regime-switching allocation across equities, bonds, and cash | Converts regime state into asset allocation or strategy-selection policy | Download NBER PDF |
| 3 | [Market regime detection via realized covariances](https://ideas.repec.org/a/eee/ecmode/v111y2022ics0264999322000785.html), Bucci and Ciciretti, 2022 | Economic Modelling / RePEc | Realized covariance regime detection with hierarchical clustering and nonlinear models | Volatility regime, correlation stress, market-wide risk state | Download from publisher if accessible; arXiv preprint: https://arxiv.org/abs/2104.03667 |
| 4 | [Clustering Market Regimes Using the Wasserstein Distance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905), Blanka Horvath, Zacharia Issa, and Aitor Muguruza, 2021 | SSRN / arXiv | Wasserstein k-means over empirical return distributions | Nonparametric regime buckets for direction, slope, volatility, and distribution-shape labels | Already local: `TraderLab/research/papers/ssrn-3947905.pdf`; arXiv: https://arxiv.org/abs/2110.11848 |
| 5 | [Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures](https://arxiv.org/abs/2306.15835), Zacharia Issa and Blanka Horvath, 2023 | arXiv | Online MMD/signature two-sample detection and clustering | Online regime-change detector suitable for walk-forward tuning windows and crypto/equity baskets | Download arXiv PDF |
| 6 | [A Hybrid Learning Approach to Detecting Regime Switches in Financial Markets](https://arxiv.org/abs/2108.05801), Peter Akioyamen, Yi Zhou Tang, and Hussien Hussien, 2021 | arXiv / ICAIF 2020 | PCA, k-means, classification, and strategy backtests | Practical benchmark for regime labels that feed strategy selection | Download arXiv PDF |
| 7 | [Predicting Risk-adjusted Returns using an Asset Independent Regime-switching Model](https://ideas.repec.org/p/arx/papers/2107.05535.html), Nicklas Werge, 2021 | RePEc / arXiv | Sticky HMM features for bull, bear, and high-volatility states | Direction, volatility, turnover control, and regime persistence | Download arXiv PDF |
| 8 | [Market Regime Identification Using Hidden Markov Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3406068), Yuan Yuan and Gautam Mitra, 2016 | SSRN | HMM for FTSE 100 and Euro Stoxx 50 market regimes | Baseline HMM implementation and state interpretation | Download SSRN PDF |
| 9 | [Asymmetric Hidden Markov Modeling of Order Flow Imbalances for Microstructure-Aware Market Regime Detection](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5315733), Jay Salvi, 2025 | SSRN | Asymmetric HMM on order-flow imbalance | Volume/orderbook regime labels and intraday strategy selection | Inspect as applied microstructure seed |

## BFS Corpus Strategy

Depth 0 seeds:

- all local PDFs under `TraderLab/research/papers/`
- the nine curated papers above
- the generated scout list at `TraderOptimizer/reports/regime_detection_paper_lookup_20260531_candidates.jsonl`

Depth 1 expansion:

- for each seed, collect references and citing papers from RePEc/IDEAS, arXiv linked tools, Semantic Scholar, OpenAlex, or publisher pages.
- keep the top 2 neighbors by observed source h-index first, then max author h-index, then citation count, then direct Trader regime relevance.
- record the ranking metric actually observed. If only citations are available, do not label it as h-index.

Depth 2 expansion:

- expand only papers that improve one of the five Trader regime dimensions or provide an implementation/validation template.
- reject papers with full-sample labels only, no walk-forward setup, or no way to avoid leakage.

Required corpus artifacts:

- `paper_nodes.jsonl`
- `edges.jsonl`
- `download_manifest.jsonl`
- `to_lookup_note.md`
- `corpus_manifest.json`

The reusable agent is installed at:

- `/Users/vrajpandya/.codex/skills/trader-regime-paper-scout`
- `/Users/vrajpandya/.codex/skills/trader-research-corpus-creator`

## Strategy-Regime Representation Notes

Use papers to define market-region features, then attach exact region IDs to
strategy tuning profiles:

| Trader dimension | Paper families to use | Example config meaning |
| --- | --- | --- |
| Direction and slope severity | HMM, change-point, Wasserstein distribution clusters, momentum papers | "Tuned for upward, severity 3 slope, persistent state" |
| Volatility | Markov-switching volatility, realized covariance, covariance stress | "Tuned for high individual volatility and stressed market covariance" |
| Index futures direction | cross-asset HMM, asset-allocation regimes, index/futures lead-lag papers | "Tuned when futures confirm or conflict with equity direction" |
| Options probability map | option-implied regime and volatility-surface papers to add in BFS | "Tuned for skew/term-structure momentum and probability mass shift" |
| Volume/orderbook | order-flow HMM and microstructure regime papers | "Tuned for high-volume directional imbalance with pending L2 book pressure" |

Advanced parameters are now tracked in
`TraderOptimizer/reports/regime_corpus_bfs_20260531/advanced_regime_parameters.md`.
They are not promotion gates; they are richer regime descriptors for later
matching between optimizer runs, generated regions, and strategy configs.

Open research gaps:

- Need a dedicated options-regime seed set.
- Need source h-index or author h-index lookup for SSRN/arXiv-only papers, where no journal source h-index exists.
- Need an implementation note mapping online/causal regime detection to TraderOptimizer train/validation windows.
