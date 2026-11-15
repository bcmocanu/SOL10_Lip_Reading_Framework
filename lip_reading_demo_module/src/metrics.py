"""Metricile originale ale evaluatorului DB (protocol pastrat integral).

WER/CER sunt calculate pentru fiecare video si apoi mediate.
Normalizarea originala elimina diacriticele si punctuatia.
"""

import re
import unicodedata

from bert_score import score
from rouge_score import rouge_scorer

def levenshtein_distance(seq1, seq2):
    m, n = (len(seq1), len(seq2))
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def wer(reference, hypothesis):
    r = reference.strip().split()
    h = hypothesis.strip().split()
    return levenshtein_distance(r, h) / max(len(r), 1)

def cer(reference, hypothesis):
    r = list(reference.strip())
    h = list(hypothesis.strip())
    return levenshtein_distance(r, h) / max(len(r), 1)

def evaluate_metrics(groundtruth, predictie):
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_scores = rouge.score(groundtruth, predictie)
    return {
        'WER': wer(groundtruth, predictie),
        'CER': cer(groundtruth, predictie),
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
    }

def compute_scores(references, predictions, bert_score_path):
    P, R, F1 = score(
        predictions,
        references,
        lang='ro',
        num_layers=8,
        model_type=bert_score_path,
        rescale_with_baseline=False,
    )
    global_scores = {}
    for idx in range(len(references)):
        g = references[idx]
        p = predictions[idx]
        metrics = evaluate_metrics(g, p)
        metrics['BERTScore-F1'] = F1[idx].item()
        metrics['BERTScore-P'] = P[idx].item()
        metrics['BERTScore-R'] = R[idx].item()
        for k, v in metrics.items():
            global_scores[k] = global_scores.get(k, 0) + v
    count = len(references)
    global_scores = {k: round(v / count, 4) for k, v in global_scores.items()}
    return global_scores

def normalizeaza_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join((c for c in text if unicodedata.category(c) != 'Mn'))
    text = re.sub('[^a-z0-9]', ' ', text)
    text = re.sub('\\s+', ' ', text).strip()
    return text
