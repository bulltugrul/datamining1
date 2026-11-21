"""Öğrenci göstergeleri ve gruplama sonuçlarının gerçek çıktılarından grafik üret."""

import json
import os
from pathlib import Path
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def save(fig, name):
    directory = ROOT / "outputs/figures"
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=170)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    meta = json.loads((ROOT / "outputs/patterns/metadata.json").read_text(encoding="utf-8"))
    session = json.loads((ROOT / "outputs/replay/session.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "outputs/patterns/training_snapshots.npz") as data:
        features, labels = data["features"], data["cluster"]
    palette = ["#2463a6", "#c56b14", "#22846b", "#845da3", "#536878", "#b14e6f"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    candidates = meta["candidates"]
    axes[0].plot([row["clusters"] for row in candidates], [row["silhouette"] for row in candidates],
                 color="#52647c", marker="o")
    axes[0].scatter([meta["selected_clusters"]], [meta["selected_silhouette"]],
                    s=160, facecolors="none", edgecolors="#c56b14", linewidths=2)
    axes[0].set(title="Grup sayısının seçimi", xlabel="Grup sayısı", ylabel="Siluet değeri")
    axes[0].set_xticks([row["clusters"] for row in candidates])
    axes[0].grid(alpha=.2)
    for cluster in range(meta["selected_clusters"]):
        selected = labels == cluster
        axes[1].scatter(features[selected, 0], features[selected, 1], s=8, alpha=.15,
                        color=palette[cluster], label=f"Grup {cluster}")
    axes[1].set(title="Dört özellikli gruplamanın iki özellikte görünümü", xlabel="Yakın dönem doğru cevap oranı",
                 ylabel="Aktif becerilerin referans tahmin ortalaması", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    axes[1].legend()
    axes[1].grid(alpha=.2)
    save(fig, "pattern_selection_and_groups")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), layout="constrained")
    fields = [("mean_observed_accuracy", "Doğru cevap oranı", 100, "Yüzde"),
              ("mean_reference_probability", "Referans başarı tahmini", 100, "Yüzde"),
              ("median_response_seconds", "Yanıt süresi", 1, "Saniye"),
              ("mean_hints", "İpucu kullanımı", 1, "Cevap başına ortalama")]
    for ax, (field, title, multiplier, unit) in zip(axes, fields):
        profiles = meta["profiles"]
        ax.bar([f"Grup {row['cluster']}" for row in profiles], [row[field]*multiplier for row in profiles],
               color=palette[:len(profiles)])
        ax.set(title=title, ylabel=unit)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Eğitim geçmişlerinde bulunan davranış grupları", fontsize=14)
    save(fig, "pattern_profiles")

    states = [session["initial"]] + [event["after"] for event in session["events"]]
    skills = states[-1]["skill_scores"]
    rows = int(np.ceil(len(skills)/2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3*rows), layout="constrained", sharex=True, sharey=True,
                             squeeze=False)
    for ax, skill in zip(axes.flat, skills):
        probability, observed = [], []
        for state in states:
            item = next((item for item in state["skill_scores"] if item["skill"] == skill["skill"]), None)
            probability.append(item["predicted_success_percent"] if item else np.nan)
            observed.append(item["observed_correct_percent"] if item and item["observed_correct_percent"] is not None else np.nan)
        ax.plot(range(len(states)), probability, color="#2463a6", marker="o", markersize=3,
                label="Sabit referans tahmini")
        ax.plot(range(len(states)), observed, color="#c56b14", linestyle="--", marker=".",
                label="Son 20 etkileşimde gözlenen")
        ax.set(title="\n".join(textwrap.wrap(skill["name"], 34)), xlabel="Yeniden oynatım adımı", ylabel="Yüzde",
               ylim=(-3, 103), xlim=(0, len(states)-1))
        ax.set_xticks(range(0, len(states), 5))
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(skills):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Beceri bazında tahmini başarı ve gözlenen doğru cevap oranı", fontsize=14)
    save(fig, "replay_skill_progress")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained", sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    steps = [event["step"] for event in session["events"]]
    axes[0].plot(steps, [100*event["predicted_before_response"] for event in session["events"]],
                 marker="o", color="#2463a6", label="Cevap görülmeden başarı tahmini")
    axes[0].scatter(steps, [100*event["observed_correct"] for event in session["events"]],
                    color="#22846b", marker="x", s=50, label="Gerçek cevap sonucu 0 veya 100")
    axes[0].set(ylabel="Yüzde", ylim=(-5, 105), title="Her cevap öncesindeki tahmin ve gerçekleşen cevap")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=.2)
    groups = [state["behavior"]["cluster"] for state in states]
    axes[1].step(range(len(states)), groups, where="post", color="#845da3")
    axes[1].scatter(range(len(states)), groups, color="#845da3", s=20)
    axes[1].set(xlabel="Yeniden oynatım adımı", ylabel="Davranış grubu", title="Güncel geçmişin en yakın grubu")
    axes[1].set_yticks(range(meta["selected_clusters"]), [f"Grup {i}" for i in range(meta["selected_clusters"])])
    axes[1].set_xticks(range(0, len(states), 5))
    axes[1].grid(alpha=.2)
    save(fig, "replay_predictions_and_groups")

    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    recommendations = [state["recommendation"] for state in states]
    ax.step(range(len(states)), [100*row["target_probability"] for row in recommendations],
             where="post", color="#c56b14", label="Seçim kuralının hedeflediği başarı")
    ax.plot(range(len(states)), [100*row["predicted_probability"] for row in recommendations],
             color="#2463a6", marker="o", markersize=4, label="Önerilen sorunun tahmini başarısı")
    ax.set(xlabel="Yeniden oynatım adımı", ylabel="Yüzde", ylim=(0, 100),
           title="Güncel geçmişe göre değişen soru önerisi hedefi")
    ax.set_xticks(range(0, len(states), 5))
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "replay_recommendation_targets")
    print("Tracking figures generated from saved pattern and replay outputs.")


if __name__ == "__main__":
    main()
