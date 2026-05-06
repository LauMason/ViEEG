import torch


# ── 情绪分类用 ────────────────────────────────────────────────────────────────

def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


# ── 脑解码：基础 Top-K 检索 ───────────────────────────────────────────────────

def topk_retrieval_accuracy(eeg_features, img_centers, labels, ks=(1, 3, 5)):
    """
    eeg_features : (N, D) L2-normalised
    img_centers  : (C, D) L2-normalised
    labels       : (N,)   int64, values in [0, C)
    Returns dict {k: float}
    """
    sim = eeg_features @ img_centers.t()  # (N, C)
    max_k = max(ks)
    _, topk_idx = sim.topk(max_k, dim=1)  # (N, max_k)
    labels_col = labels.view(-1, 1)
    return {k: (topk_idx[:, :k] == labels_col).any(dim=1).float().mean().item()
            for k in ks}


# ── 脑解码：ViEEG 三路 ablation 检索 ─────────────────────────────────────────

def vieeg_ablation_topk(
        v_eeg, m_eeg, b_eeg,  # (N, D) each, L2-normalised EEG features
        v_img, m_img, b_img,  # (C, D) each, L2-normalised image features
        labels,  # (N,)
        ks=(1, 3, 5),
):
    """
    Compute Top-K retrieval for 6 ablation combinations:
      0: visual only     (v_eeg  ↔ v_img)
      1: mask only       (m_eeg  ↔ m_img)
      2: mask01 only     (b_eeg  ↔ b_img)
      3: visual+mask     (vm_eeg ↔ vm_img)
      4: visual+mask01   (vb_eeg ↔ vb_img)
      5: mask+mask01     (mb_eeg ↔ mb_img)

    Returns dict:
      {
        "visual":        {1: x, 3: x, 5: x},
        "mask":          {1: x, 3: x, 5: x},
        "mask01":        {1: x, 3: x, 5: x},
        "visual+mask":   {1: x, 3: x, 5: x},
        "visual+mask01": {1: x, 3: x, 5: x},
        "mask+mask01":   {1: x, 3: x, 5: x},
      }
    """
    pairs = {
        "visual": (v_eeg, v_img),
        "mask": (m_eeg, m_img),
        "mask01": (b_eeg, b_img),
        "visual+mask": (torch.cat([v_eeg, m_eeg], 1), torch.cat([v_img, m_img], 1)),
        "visual+mask01": (torch.cat([v_eeg, b_eeg], 1), torch.cat([v_img, b_img], 1)),
        "mask+mask01": (torch.cat([m_eeg, b_eeg], 1), torch.cat([m_img, b_img], 1)),
    }

    results = {}
    for name, (eeg_feat, img_feat) in pairs.items():
        # re-normalise concatenated features
        eeg_n = eeg_feat / eeg_feat.norm(dim=1, keepdim=True)
        img_n = img_feat / img_feat.norm(dim=1, keepdim=True)
        results[name] = topk_retrieval_accuracy(eeg_n, img_n, labels, ks=ks)

    return results
