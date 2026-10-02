# BiTransAct

**A Bidirectional Cross-Attention Transformer with Evidential Uncertainty Estimation for MicroRNA–mRNA Target Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Source code, data and trained model for the paper *"BiTransAct: A Bidirectional Cross-Attention Transformer with Evidential Uncertainty Estimation for MicroRNA–mRNA Target Prediction"* (Babagolzadeh, Mirzarezaee, Sadeghi; submitted to *BMC Research Notes*).

BiTransAct predicts whether a microRNA (miRNA) functionally targets a given mRNA. Separate transformer encoders read the miRNA and the candidate binding site, bidirectional cross-attention lets each representation query the other, and the fused representation is classified at the site level. Site-level scores are then aggregated over a sliding window to give a gene-level call.

The repository contains two variants:

| Variant | File | Output head |
|---|---|---|
| **BiTransAct** | `src/BiTransAct_Model.py` | Softmax probability |
| **BiTransAct-DST** | `src/BiTransAct_DST_Model.py` | Dempster–Shafer belief plus an uncertainty score |

---

## Method at a glance

**Input.** The miRNA (padded to 30 nt) and each candidate site (40 nt) are encoded as the sum of three embeddings: a 64-dimensional nucleotide embedding (alphabet `A, C, G, U, X`, with `X` as padding), a learned positional embedding, and a pairing embedding with three states (mismatch/gap, G–U wobble, Watson–Crick). The pairing states come from a Smith–Waterman local alignment of the extended miRNA seed region against the site. This pairing representation and the benchmark protocol are taken from [Mimosa](#references); the contribution of this work is the interaction stage. The mRNA is reversed so that both sequences share the same 5′→3′ orientation relative to duplex formation.

**BiTransAct.**

- Two separate encoder stacks (miRNA and site), 16 layers each, 8 attention heads, hidden size 64, dropout 0.1.
- Bidirectional cross-attention: site→miRNA and miRNA→site.
- Mean pooling, concatenation, a linear fusion layer with ReLU, and a two-layer softmax classifier.

**BiTransAct-DST.** Same encoders and cross-attention, but the classification head outputs non-negative evidence that parameterizes a Dirichlet distribution. Class belief is the evidence divided by the total strength, and the remaining mass is reported as uncertainty (following Sensoy et al.'s evidential deep learning). Calibration of these outputs was not independently evaluated.

**Gene-level inference.** Each 3′-UTR is scanned with overlapping 40-nt windows (step 5 nt).

- *BiTransAct:* a gene is called a target if at least one window scores above 0.5 (equivalent to max-pooling the site probabilities).
- *BiTransAct-DST:* per-window belief and uncertainty are averaged (mean-belief rule), and the class with the larger mean belief is returned together with the mean uncertainty.

**Training.** Adam, learning rate 1e-4, weight decay 1e-5, batch size 256. The best checkpoint on the validation set is kept. The released BiTransAct checkpoint was selected at epoch 24.

---

## Repository structure

```
BiTransAct/
├── src/
│   ├── BiTransAct_Model.py        # BiTransAct: training and gene-level evaluation
│   ├── BiTransAct_DST_Model.py    # BiTransAct-DST (evidential head)
│   └── utils.py                   # data loading, Smith–Waterman pairing, metrics
├── data/
│   ├── site_level/
│   │   └── miRAW_Train_Validation.txt          # site-level train / validation pairs
│   └── gene_level_test_sets/
│       └── miRAW_Test0.txt … miRAW_Test9.txt   # ten gene-level test sets
├── checkpoints/
│   └── best_model.pth             # trained BiTransAct (16 encoder layers)
├── LICENSE
└── README.md
```

## Data

The experiments use the public **Mimosa benchmark** (Bi *et al.*, 2024), assembled from experimentally supported interaction resources (DIANA-TarBase, miRTarBase) together with PAR-CLIP and CLASH evidence. The partitions are used as released, so results are directly comparable with prior work. The data files are redistributed here for convenience; please cite the original benchmark and follow its terms of use.

- **Site level (training / validation):** `miRAW_Train_Validation.txt`. Training: 26,995 positive and 27,469 negative examples. Validation: 2,193 positive and 2,136 negative. Columns: `mirna_id`, `mirna_seq`, `mrna_id`, `mrna_seq`, `label`, `split`.
- **Gene level (test):** ten independent balanced test sets, each with 548 positive and 548 negative pairs. Columns: `mirna_id`, `mirna_seq`, `mrna_id`, `mrna_seq`, `label`. The negative pairs are the same in all ten sets; the positive pairs differ.

`label = 1` marks a functional interaction and `0` a non-interaction.

## Installation

```bash
git clone https://github.com/MrGolzadeh/BiTransAct.git
cd BiTransAct
pip install torch numpy scikit-learn
```

The code runs on CPU; a GPU is not required for evaluation.

## Usage

The scripts read their input files from the **current working directory**, so first copy the data and checkpoint next to the scripts.

**Evaluate the provided BiTransAct model on the ten test sets**

```bash
cd src
cp ../data/site_level/miRAW_Train_Validation.txt .
cp ../data/gene_level_test_sets/miRAW_Test*.txt .
cp ../checkpoints/best_model.pth .
python *BiTransAct_Model.py
```

On Windows, use `copy` instead of `cp`. For each test set the script prints accuracy, precision, recall, specificity, F1 and NPV. The Smith–Waterman alignment runs in pure Python, so evaluation takes a while.

> **Note on the file name.** The name of `BiTransAct_Model.py` begins with an invisible character. The wildcard in `python *BiTransAct_Model.py` handles this on Linux, macOS and Git Bash; in other shells, press Tab to auto-complete the name.

**Train BiTransAct from scratch.** In `BiTransAct_Model.py`, uncomment `perform_train()` in the `__main__` block and run the script as above. The best checkpoint is written to `best_model.pth` in the working directory.

**Train and evaluate BiTransAct-DST**

```bash
python BiTransAct_DST_Model.py
```

This trains the evidential model and then evaluates it on the ten test sets, printing the usual metrics plus the average uncertainty. Both scripts write to the same file name, so run DST in a separate folder if you want to keep the provided `best_model.pth`.

## Results

**BiTransAct, gene level** (mean ± SD over the ten test sets; benchmark results of the other methods are those reported in the Mimosa study):

| Model | Accuracy | Precision† | Recall | Specificity | F1 | NPV |
|---|---|---|---|---|---|---|
| deepTarget | 0.65 | 0.83 | 0.34 | 0.93 | 0.49 | 0.60 |
| miRAW | 0.70 | 0.67 | 0.79 | 0.61 | 0.72 | 0.74 |
| TargetNet | 0.72 | 0.65 | 0.94 | 0.50 | 0.77 | 0.89 |
| Mimosa | 0.75 | 0.67 | 0.92 | 0.58 | 0.79 | 0.89 |
| **BiTransAct** | 0.7528 ± 0.0059 | 0.6731 ± 0.0047 | 0.9382 ± 0.0118 | 0.5675 ± 0.0000 | 0.7914 ± 0.0060 | 0.9020 ± 0.0169 |

Compared with Mimosa, recall and NPV are higher by 0.0088 and 0.0100, accuracy, precision and F1 are essentially unchanged, and specificity is lower. None of these differences is statistically significant (paired t-test over ten test sets: recall *p* = 0.236, NPV *p* = 0.318), so they should be read as descriptive differences. BiTransAct has the best mean rank across the six metrics (2.06 versus 2.31 for Mimosa). The specificity has no spread because the negative pairs are identical in every test set.

**Effect of encoder depth** (BiTransAct, mean over the ten test sets; this is Table S1, Additional file 1 of the paper). "Best epoch" is the epoch of minimum validation loss.

| Depth | Best epoch | Accuracy | Precision† | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|
| 8 layers | 22 | 0.748 | 0.670 | 0.920 | 0.577 | 0.785 |
| **16 layers** | 24 | 0.753 | 0.673 | 0.938 | 0.568 | 0.791 |
| 26 layers | 57 | 0.692 | 0.620 | 0.945 | 0.438 | 0.754 |

Going from 8 to 16 layers improved accuracy, recall and F1; 26 layers gave only a small recall gain while specificity, accuracy and F1 dropped. The 16-layer model was therefore retained.

**BiTransAct-DST.** At its best checkpoint, DST reached site-level accuracy 0.7588 and F1 0.7342. With mean-belief aggregation at gene level:

| Accuracy | Precision† | Recall | Specificity | F1 | NPV | Avg. uncertainty |
|---|---|---|---|---|---|---|
| 0.5878 ± 0.0109 | 0.5646 ± 0.0094 | 0.2741 ± 0.0217 | 0.9015 ± 0.0000 | 0.3990 ± 0.0253 | 0.5540 ± 0.0074 | 0.0339 ± 0.0001 |

The two variants therefore serve different purposes: BiTransAct is a high-recall candidate generator, whereas BiTransAct-DST (mean-belief) is a specificity-oriented negative-screening option. Uncertainty-aware aggregation, such as excluding high-uncertainty windows, has not been evaluated yet.

† Precision is computed with scikit-learn's `average_precision_score` on the hard class predictions, as in the evaluation code of this repository. It is not identical to TP / (TP + FP).

## Limitations
-Although BiTransAct was designed and evaluated using bidirectional cross-attention, the isolated effect of bidirectional cross-attention was not examined through a controlled ablation against an otherwise identical BiTransAct variant using unidirectional cross-attention. Therefore, the performance difference between BiTransAct and Mimosa cannot be attributed solely to the bidirectionality of cross-attention.
- The two variants were trained under different schedules, so the effect of the evidential head is not isolated.
- The benchmark covers experimentally supported human interactions only; other species are untested.
- The model uses linear sequence and local-alignment pairing signals; it does not use tertiary structure, protein cofactors or tissue-specific expression.
- Annotation noise and class imbalance in the source databases may influence the metrics.

## Citation

If you use this code, please cite the paper (details will be updated after publication):

```bibtex
@unpublished{babagolzadeh2026bitransact,
  title  = {BiTransAct: A Bidirectional Cross-Attention Transformer with Evidential Uncertainty Estimation for MicroRNA--mRNA Target Prediction},
  author = {Babagolzadeh, Saeed and Mirzarezaee, Mitra and Sadeghi, Mehdi},
  note   = {Submitted to BMC Research Notes},
  year   = {2026}
}
```

## References

1. Bi Y, Li F, Wang C, *et al.* Advancing microRNA target site prediction with transformer and base-pairing patterns (Mimosa). *Nucleic Acids Res.* 2024;52(19):11455–11465. doi:10.1093/nar/gkae782
2. Pla A, Zhong X, Rayner S. miRAW: A deep learning-based approach to predict microRNA targets by analyzing whole microRNA transcripts. *PLoS Comput Biol.* 2018;14(7):e1006185. doi:10.1371/journal.pcbi.1006185
3. Min S, Lee B, Yoon S. TargetNet: functional microRNA target prediction with deep neural networks. *Bioinformatics.* 2022;38(3):671–677. doi:10.1093/bioinformatics/btab733
4. Lee B, Baek J, Park S, Yoon S. deepTarget: End-to-end learning framework for microRNA target prediction using deep recurrent neural networks. *ACM-BCB* 2016:434–442. doi:10.1145/2975167.2975212
5. Sensoy M, Kaplan L, Kandemir M. Evidential deep learning to quantify classification uncertainty. *NeurIPS* 2018;31.
6. Vaswani A, *et al.* Attention is all you need. *NeurIPS* 2017;30:5998–6008.

## Acknowledgements

Computational resources were provided by the Institute for Research in Fundamental Sciences (IPM). We thank Hamed Jamshidi Moghadam for technical advice on parts of the implementation.

## License

The code is released under the [MIT License](LICENSE). The data files originate from the Mimosa benchmark and remain subject to its terms.

## Contact

Saeed Babagolzadeh: [GitHub @MrGolzadeh](https://github.com/MrGolzadeh) · Corresponding author: Mitra Mirzarezaee (mirzarezaee@iau.ac.ir)
