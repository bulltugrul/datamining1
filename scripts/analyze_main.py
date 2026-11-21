"""Tamamlanmış dokuz ana deneyi denetle, özetle ve görselleştir."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from kt_project.experiment_results import METHODS, SEEDS, audit_main_run, select_prototype

RUN = ROOT / "outputs/main/three_seed_v1"
LABELS = {"baseline": "Promptsuz uyarlama", "kpt": "KPT", "kmpt": "KMPT"}
COLORS = {"baseline": "#52647c", "kpt": "#2463a6", "kmpt": "#c56b14"}


def save_figure(fig, name):
    folder = ROOT / "outputs/figures"
    folder.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=170)
    plt.close(fig)


def figures(results, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ("auc", "accuracy", "bce"),
                                  ("Test ROC AUC", "Test doğruluğu", "Test BCE")):
        for x, method in enumerate(METHODS):
            item = summary[method]["test"][metric]
            values = [next(row["test"][metric] for row in results["runs"]
                           if row["method"] == method and row["seed"] == seed) for seed in SEEDS]
            ax.scatter(x+np.array([-.08, 0, .08]), values, color=COLORS[method], alpha=.55, s=30)
            ax.errorbar(x, item["mean"], yerr=item["sample_sd"], fmt="o", color=COLORS[method],
                        capsize=7, markersize=7, linewidth=2)
        ax.set_xticks(range(3), ["Promptsuz", "KPT", "KMPT"])
        ax.set_title(title)
        ax.set_ylabel("Üç başlangıcın ortalaması ve örnek standart sapması")
        ax.grid(axis="y", alpha=.2)
        ax.margins(x=.25, y=.25)
    fig.suptitle("Bağımsız testte üç başlangıçlı ana karşılaştırma", fontsize=15)
    save_figure(fig, "main_test_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained", sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for method in METHODS:
            rows = json.loads((RUN / f"seed_{seed}/{method}_history.json").read_text())
            selected = next(row for row in results["runs"] if row["method"] == method and row["seed"] == seed)
            ax.plot([row["epoch"] for row in rows], [row["validation"]["bce"] for row in rows],
                    marker="o", markersize=3, color=COLORS[method], label=LABELS[method])
            ax.scatter([selected["best_epoch"]], [selected["validation"]["bce"]],
                       facecolors="none", edgecolors=COLORS[method], s=90, linewidths=2)
        ax.set(title=f"Başlangıç {seed}", xlabel="Dönem")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Doğrulama cevap kaybı BCE")
    axes[-1].legend(fontsize=9)
    fig.suptitle("Doğrulama kaybı ve seçilen dönemler", fontsize=15)
    save_figure(fig, "main_validation_curves")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained", sharex=True, sharey=True)
    palette = ["#2463a6", "#c56b14", "#22846b"]
    for ax, method in zip(axes, METHODS):
        for seed, color in zip(SEEDS, palette):
            with np.load(RUN / f"seed_{seed}/{method}_test_predictions.npz", allow_pickle=False) as saved:
                fpr, tpr, _ = roc_curve(saved["label"], saved["probability"])
            score = next(row["test"]["auc"] for row in results["runs"]
                         if row["method"] == method and row["seed"] == seed)
            ax.plot(fpr, tpr, color=color, label=f"{seed}: AUC {score:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#8894a3", linewidth=1)
        ax.set(title=LABELS[method], xlabel="Yanlış pozitif oranı", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=8.5)
    axes[0].set_ylabel("Doğru pozitif oranı")
    fig.suptitle("Aynı bağımsız test hedeflerinde ROC eğrileri", fontsize=15)
    save_figure(fig, "main_roc_curves")


def write_report(results, summary, integrity, selection):
    lines = ["# Üç Başlangıçlı Ana Deney Sonuçları", "",
             "Promptsuz uyarlama, KPT ve KMPT; aynı öğrenci ayrımında, her başlangıç için ortak LKM ağırlıklarından başlayarak karşılaştırıldı. En iyi dönem yalnızca doğrulama BCE değerinden seçildi. Seçilen ağırlıklar bağımsız testte değerlendirildi.", "",
             "## Veri ve deney düzeni", "",
             f"Her koşul {results['runs'][0]['training_target_count']} eğitim ve {results['runs'][0]['validation_target_count']} doğrulama hedefi kullandı. Testte {integrity['test_students']} öğrenciye ait {integrity['test_targets']} cevap hedefi bulunuyor. Her öğrencinin ilk etkileşimi başlangıç geçmişi olarak kullanıldığı için hedef sayıları ham etkileşim sayılarından daha düşüktür.", "",
             "Başlangıçlar 17, 42 ve 2026. Aynı başlangıç içindeki üç yöntemin LKM içerik özetleri eşleşiyor. Farklı başlangıçların LKM özetleri farklı. Veri hazırlama ve model kaynaklarının eğitim anındaki içerik özetleri doğrulandı.", "",
             "## Birincil test sonuçları", "",
             "Değerler üç başlangıcın ortalaması ve örnek standart sapmasıdır. AUC ile doğrulukta yüksek, BCE değerinde düşük sonuç daha iyidir. Ölçüler etkileşim ağırlıklıdır; öğrenci başına eşit ağırlıklı ortalamalar değildir.", "",
             "| Yöntem | Test AUC | Test doğruluğu yüzde | Test BCE |",
             "|---|---|---|---|"]
    for method in METHODS:
        stats = summary[method]["test"]
        lines.append(f"| {LABELS[method]} | {stats['auc']['mean']:.4f} ± {stats['auc']['sample_sd']:.4f} | "
                     f"{100*stats['accuracy']['mean']:.2f} ± {100*stats['accuracy']['sample_sd']:.2f} | "
                     f"{stats['bce']['mean']:.4f} ± {stats['bce']['sample_sd']:.4f} |")
    lines += ["", "Bu standart sapma, sabit bir öğrenci ayrımında başlangıçlar arasındaki değişkenliği gösterir. Öğrenci evrenine ilişkin güven aralığı veya istatistiksel anlamlılık sonucu değildir.", "",
              "## Her başlangıcın sonucu", "",
              "| Başlangıç | Yöntem | Seçilen dönem | Doğrulama BCE | Test AUC | Test BCE | Test doğruluğu yüzde |",
              "|---|---|---|---|---|---|---|"]
    for seed in SEEDS:
        for method in METHODS:
            row = next(row for row in results["runs"] if row["method"] == method and row["seed"] == seed)
            lines.append(f"| {seed} | {LABELS[method]} | {row['best_epoch']} | {row['validation']['bce']:.4f} | "
                         f"{row['test']['auc']:.4f} | {row['test']['bce']:.4f} | {100*row['test']['accuracy']:.2f} |")
    best = max(METHODS, key=lambda method: summary[method]["test"]["auc"]["mean"])
    lines += ["", "## Sonuçların yorumu", "",
              f"Bu uygulama ve sabit veri ayrımında en yüksek ortalama test AUC değeri {LABELS[best]} yönteminde elde edildi."]
    for method in ("kpt", "kmpt"):
        difference = summary[method]["auc_difference_from_baseline"]
        lines += ["", f"{LABELS[method]} ile promptsuz uyarlama arasındaki aynı başlangıçlar üzerinden hesaplanmış AUC farkı "
                       f"{difference['mean']:+.4f}; bu farkın örnek standart sapması {difference['sample_sd']:.4f}."]
    lines += ["", "Karşılaştırma yöntemlerin toplam etkisini ölçer. Şablonlar, PromEnc, başlık kullanımı ve KMPT'nin yardımcı görevi birlikte değişir. Sonuç yalnızca soft promptlara atfedilemez. Üç başlangıç ve tek öğrenci ayrımı yöntemin tüm veri kümelerinde üstünlüğünü kanıtlamaz.", "",
              "Model kapasitesi, çok becerili kayıtların korunması, dikkat düzeni ve veri ayrımı makaledeki deneyin birebir kopyası değildir. Bu farklılıkların tekil etkileri burada ayrı ablasyonlarla ölçülmedi.", "",
              "## Eğitimde görülmeyen sorular", "",
              f"Test hedeflerinin {integrity['unseen_question_test_targets']} tanesinde eğitimde görülmeyen soru kimliği var. Bu hedefler birincil değerlendirmeden çıkarılmadı. Bilinen soru alt grubunun AUC sonuçları aşağıdadır.", "",
              "| Yöntem | Bilinen sorularda AUC ortalama ve standart sapma |", "|---|---|"]
    for method in METHODS:
        values = [row["test"]["known_question_subset"]["auc"] for row in results["runs"] if row["method"] == method]
        lines.append(f"| {LABELS[method]} | {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f} |")
    lines += ["", "## Örnek oturumda kullanılacak model", "",
              f"Örnek oturum için önceden belirlenen yöntem KMPT'dir. Üç KMPT ağırlığı içinden yalnızca doğrulama BCE değeriyle başlangıç {selection['seed']} seçildi. Bu seçimde test skorları kullanılmadı. Bu tercih, KMPT'nin testte en iyi yöntem olduğu iddiası değildir.", "",
              "## Çalışma maliyeti ve kayıtlar", "",
              f"Kayıtlı toplam deney süresi {results['total_seconds']/60:.2f} dakika; en yüksek ayrılan PyTorch GPU belleği {results['peak_allocated_gpu_mb']:.1f} MiB. Aygıt {results['gpu']}, hassasiyet {results['precision']}. Süreye uyarlama dönemlerindeki doğrulama işlemleri dahildir.", "",
              "Ayarlar, ağırlıklar, dönem geçmişleri ve örnek bazındaki test tahminleri outputs/main/three_seed_v1 altında saklanır. Yeniden hesaplanan ölçüler ve örnek eşleşme kontrolleri aynı klasörün analysis bölümündedir."]
    (ROOT / "docs/ana_deney_sonuclari.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    results, summary, integrity = audit_main_run(ROOT, RUN)
    policy = json.loads((ROOT / "configs/reporting_policy.json").read_text(encoding="utf-8"))
    selection = select_prototype(results["runs"], policy)
    selection["checkpoint"] = f"outputs/main/three_seed_v1/seed_{selection['seed']}/kmpt_best.pt"
    output = RUN / "analysis"
    output.mkdir(exist_ok=True)
    for name, value in (("summary.json", summary), ("integrity.json", integrity),
                         ("prototype_selection.json", selection)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    figures(results, summary)
    write_report(results, summary, integrity, selection)
    print(json.dumps({"completed_comparisons": integrity["complete_method_seed_pairs"],
                      "test_targets": integrity["test_targets"], "summary": summary}, ensure_ascii=True))


if __name__ == "__main__":
    main()
