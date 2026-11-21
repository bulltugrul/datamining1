"""Yeni 12 koşulu denetle, ana üçlüyü ve tek yerleşim kontrolünü ayrı raporla."""

import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from kt_project.models import ModelConfig
from kt_project.plkt_training import build_tracer
from kt_project.sequences import InteractionStore, PrefixDataset, collate_prefix, to_device
from kt_project.training import cpu_state, state_digest, set_seed

RUN = ROOT / "outputs/main/paper_alignment_v2"
LABELS = {"baseline": "Promptsuz", "kpt": "KPT metin tarifi", "kmpt": "KMPT metin tarifi", "kmpt_prefix": "KMPT prefix kontrolü"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(results, config):
    assert read(RUN / "status.json")["status"] == "complete"
    assert results["test_evaluated"]
    for name, digest in results["implementation_sha256"].items():
        assert sha(ROOT / name) == digest, name
    lock = read(RUN / "test_evaluation_lock.json")
    assert lock["config_sha256"] == sha(RUN / "config.json")
    for name, digest in lock["checkpoints"].items():
        assert sha(RUN / name) == digest
    expected = {(seed, variant) for seed in config["seeds"] for variant in config["variants"]}
    assert {(row["seed"], row["variant"]) for row in results["runs"]} == expected
    assert len(results["runs"]) == len(expected)
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    expected_rows = np.array([target for _, target in PrefixDataset(store, 2, config["model"]["max_history"]).examples])
    checks = []
    for row in results["runs"]:
        folder = RUN / f"seed_{row['seed']}"
        history = read(folder / f"{row['variant']}_history.json")
        best = history[row["best_epoch"]-1]
        assert np.isclose(best["validation"]["bce"], min(item["validation"]["bce"] for item in history), atol=1e-6)
        assert np.isclose(best["validation"]["bce"], row["validation"]["bce"], atol=1e-6)
        assert row["training_target_count"] == len(PrefixDataset(store, 0, config["model"]["max_history"]))
        assert row["validation_target_count"] == len(PrefixDataset(store, 1, config["model"]["max_history"]))
        with np.load(folder / f"{row['variant']}_test_predictions.npz") as saved:
            y, p, indices = saved["label"], saved["probability"], saved["row_index"]
        np.testing.assert_array_equal(indices, expected_rows)
        np.testing.assert_array_equal(y, store.arrays["correct"][indices])
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert np.isclose(roc_auc_score(y, p), row["test"]["auc"], atol=1e-10, rtol=0)
        assert np.isclose(accuracy_score(y, p >= .5), row["test"]["accuracy"], atol=1e-10, rtol=0)
        clipped = np.clip(p.astype(float), np.finfo(float).eps, 1-np.finfo(float).eps)
        bce = float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean())
        assert np.isclose(bce, row["test"]["bce"], atol=5e-5)
        saved = torch.load(folder / f"{row['variant']}_best.pt", map_location="cpu", weights_only=True)
        assert saved["format"] == "plkt_method_v2" and saved["variant"] == row["variant"]
        assert saved["layout"] == row["layout"] and saved["config"] == results["model_config"]
        assert all(bool(torch.isfinite(value).all()) for value in saved["model_state"].values())
        checks.append({"seed": row["seed"], "variant": row["variant"], "test_targets": len(y),
                       "metrics_recomputed": True, "validation_selection_verified": True})
    for seed in config["seeds"]:
        digests = {row["initial_lkm_sha256"] for row in results["runs"] if row["seed"] == seed}
        assert len(digests) == 1
        pretrained = torch.load(RUN / f"seed_{seed}/pretrained.pt", weights_only=True, map_location="cpu")
        assert state_digest(pretrained["model_state"]) == next(iter(digests))
        original = read(RUN / f"seed_{seed}/kmpt_history.json")
        control = read(RUN / f"seed_{seed}/kmpt_prefix_history.json")
        for first, second in zip(original, control):
            assert first["response_samples"] == second["response_samples"]
            assert first["auxiliary_samples"] == second["auxiliary_samples"]
        initial_digests = []
        for variant in ("kmpt", "kmpt_prefix"):
            set_seed(seed+100)
            initial_model = build_tracer(ModelConfig(**results["model_config"]), variant, pretrained["model_state"])
            initial_digests.append(state_digest(cpu_state(initial_model.prompt_encoder)))
        assert initial_digests[0] == initial_digests[1]
    assert len({row["initial_lkm_sha256"] for row in results["runs"]}) == len(config["seeds"])
    return {"all_checks_passed": True, "conditions": len(checks), "same_test_targets": True,
            "test_targets": len(expected_rows), "test_lock_verified": True,
            "common_pretraining_per_seed_verified": True, "source_hashes_verified": True,
            "prefix_control_common_prompt_initialization_and_auxiliary_draws_verified": True, "checks": checks}


def sensitivity(results):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    config = ModelConfig(**results["model_config"])
    validation = PrefixDataset(store, 1, config.max_history)
    positions = np.linspace(0, len(validation)-1, 16, dtype=int)
    batch = to_device(collate_prefix([validation[int(i)] for i in positions]), device)
    records = []
    for row in results["runs"]:
        if row["variant"] == "baseline":
            continue
        saved = torch.load(RUN / f"seed_{row['seed']}/{row['variant']}_best.pt", weights_only=True, map_location="cpu")
        model = build_tracer(config, row["variant"]).to(device)
        model.load_state_dict(saved["model_state"])
        model.train()
        torch.manual_seed(2026)
        model.loss(batch).backward()
        gradient = model.prompt_encoder.soft_prompt.grad.detach()
        grad_l1 = float(gradient.abs().sum())
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            before = torch.sigmoid(model(batch))
            torch.manual_seed(2026)
            model.prompt_encoder.soft_prompt.add_(torch.randn_like(model.prompt_encoder.soft_prompt)*.5)
            after = torch.sigmoid(model(batch))
        delta = float((before-after).abs().max())
        records.append({"seed": row["seed"], "variant": row["variant"], "samples": len(positions),
                        "training_soft_prompt_gradient_l1": grad_l1,
                        "prediction_max_probability_delta_after_soft_prompt_perturbation": delta,
                        "inference_attention": "strict causal in both encoders", "inference_dtype": "float32"})
        if row["layout"] == "paper_tail":
            assert delta <= 1e-6
    return records


def summarize(results, config):
    summary = {}
    for variant in config["variants"]:
        rows = [row for row in results["runs"] if row["variant"] == variant]
        summary[variant] = {metric: {"mean": float(np.mean([row["test"][metric] for row in rows])),
                                     "sample_sd": float(np.std([row["test"][metric] for row in rows], ddof=1))}
                            for metric in ("auc", "accuracy", "bce")}
    return summary


def figures(results, summary, config):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    colors = ["#52647c", "#2463a6", "#c56b14", "#22846b"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for ax, metric, title in zip(axes, ["auc", "accuracy", "bce"], ["Test AUC", "Test doğruluğu", "Test BCE"]):
        for i, (variant, color) in enumerate(zip(config["variants"], colors)):
            item = summary[variant][metric]
            ax.errorbar(i, item["mean"], yerr=item["sample_sd"], fmt="o", capsize=6, color=color)
        ax.set_xticks(range(4), ["Promptsuz", "KPT\nmetin", "KMPT\nmetin", "KMPT\nprefix"])
        ax.set(title=title, ylabel="Ortalama ve başlangıçlar arası örnek standart sapması")
        ax.grid(axis="y", alpha=.2)
        ax.margins(.2, .2)
    fig.suptitle("Yöntem düzeltmesinden sonra üç başlangıçlı karşılaştırma", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT / f"outputs/figures/paper_alignment_comparison.{suffix}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained", sharey=True)
    for ax, seed in zip(axes, config["seeds"]):
        for variant, color in zip(config["variants"], colors):
            rows = read(RUN / f"seed_{seed}/{variant}_history.json")
            ax.plot([row["epoch"] for row in rows], [row["validation"]["bce"] for row in rows], label=LABELS[variant], color=color)
        ax.set(title=f"Başlangıç {seed}", xlabel="Dönem")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Nedensel tahmin kipinde doğrulama BCE")
    axes[-1].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT / f"outputs/figures/paper_alignment_validation.{suffix}", dpi=170)
    plt.close(fig)


def main():
    results, config = read(RUN / "results.json"), read(RUN / "config.json")
    integrity = audit(results, config)
    summary = summarize(results, config)
    diagnostic = sensitivity(results)
    selected = min((row for row in results["runs"] if row["variant"] == config["prototype_variant"]),
                   key=lambda row: (row["validation"]["bce"], row["seed"]))
    checkpoint = RUN / f"seed_{selected['seed']}/{selected['variant']}_best.pt"
    selection = {"format": "plkt_method_v2", "method": selected["method"], "variant": selected["variant"],
                 "layout": selected["layout"], "seed": selected["seed"], "best_epoch": selected["best_epoch"],
                 "validation_bce": selected["validation"]["bce"], "criterion": "minimum validation BCE within preselected literal KMPT",
                 "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha(checkpoint),
                 "test_metrics_used_for_selection": False}
    output = RUN / "analysis"
    output.mkdir(exist_ok=True)
    for name, value in [("summary", summary), ("integrity", integrity), ("prompt_sensitivity", diagnostic), ("prototype_selection", selection)]:
        (output / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    figures(results, summary, config)
    lines = ["# Yöntem Düzeltmesi Sonrası Deney Sonuçları", "",
        "Makaledeki şablon sırası ve eğitim/tahmin dikkat ayrımı, FBO örnekleri, RTM/FBO kayıp ölçeği ve BERT tarzı katman düzeni düzeltilerek yeni ön eğitimden başlandı. Üç başlangıçta aynı ortak ön eğitimden promptsuz, KPT ve KMPT çalıştırıldı. Dördüncü koşul yalnızca KMPT şablon yerleşimini değiştirir; bu koşul makale tarifi olarak adlandırılmaz.", "",
        "| Yöntem | Test AUC | Test doğruluğu yüzde | Test BCE |", "|---|---|---|---|"]
    for variant in config["variants"]:
        row = summary[variant]
        lines.append(f"| {LABELS[variant]} | {row['auc']['mean']:.4f} ± {row['auc']['sample_sd']:.4f} | {100*row['accuracy']['mean']:.2f} ± {100*row['accuracy']['sample_sd']:.2f} | {row['bce']['mean']:.4f} ± {row['bce']['sample_sd']:.4f} |")
    lines += ["", "Standart sapma tek öğrenci ayrımında üç başlangıç arasındadır; güven aralığı veya anlamlılık testi değildir.", "",
        "## Her koşulun doğrulama seçimi", "", "| Başlangıç | Koşul | Seçilen dönem | Doğrulama BCE | Test AUC |", "|---|---|---|---|---|"]
    for row in results["runs"]:
        lines.append(f"| {row['seed']} | {LABELS[row['variant']]} | {row['best_epoch']} | {row['validation']['bce']:.4f} | {row['test']['auc']:.4f} |")
    lines += ["", "## Parametre ve eğitim kapsamı", "",
        "| Koşul | Toplam parametre | Eğitilen parametre |", "|---|---|---|"]
    for variant in config["variants"]:
        row = next(row for row in results["runs"] if row["variant"] == variant)
        lines.append(f"| {LABELS[variant]} | {row['parameters']['total']} | {row['parameters']['trainable']} |")
    lines += ["", "LKM, ilgili ön eğitimli tahmin başlıkları ve prompt kodlayıcı birlikte güncellenir; bu saf PEFT deneyi değildir. Prefix kontrolü aynı LKM ve prompt başlangıçlarından başlar; ortak dönemlerde aynı 80/20 örnek seçimleri kullanıldığı ayrıca denetlenir."]
    lines += ["", "## Prompt duyarlılığı", "",
        "Ayrı 16 doğrulama geçmişinde eğitim kipinde prompt gradyanı, tahmin kipinde promptlara sabit büyüklükte rastgele değişiklikten sonra olasılık farkı ölçüldü. Model ağırlıkları veya kaydedilmiş kontrol noktaları bu tanısal işlemle değiştirilmedi. Katı nedensellikte sondaki promptların doğrudan etkisi yoktur; baştaki prompt kontrolü bu erişim yolunu açar. Bu bulgu makalenin metinsel tarifine ilişkindir; yazarların kodunun incelendiği iddiası değildir.", "",
        "| Başlangıç | Koşul | Eğitim prompt gradyanı L1 | Tahminde en büyük olasılık değişimi |", "|---|---|---|---|"]
    for row in diagnostic:
        lines.append(f"| {row['seed']} | {row['variant']} | {row['training_soft_prompt_gradient_l1']:.6g} | {row['prediction_max_probability_delta_after_soft_prompt_perturbation']:.6g} |")
    lines += ["", "## Kullanılan örnek model ve sınırlar", "",
        f"Önceden belirlenen KMPT metin tarifi içinde yalnızca doğrulama BCE değeriyle başlangıç {selection['seed']} seçildi. Seçim testte en yüksek AUC veren koşulu seçme işlemi değildir. Örnek oturum ve sonraki öğrenci göstergeleri bu ağırlıklara bağlanır.", "",
        f"Toplam {integrity['conditions']} koşul aynı {integrity['test_targets']} test hedefinde değerlendirildi. Tüm eğitimler bitip ayarlar ve kontrol noktaları kilitlendikten sonra test hesaplandı. Bununla birlikte bu test ayrımı eski sürümde zaten görülmüştür; yeni sayılar tamamen yeni ve dokunulmamış doğrulayıcı deney olarak sunulmaz.", "",
        "Ölçekli model kapasitesi, öğrenci bazlı 70/15/15 ayrımı, çok kavramlı tek olayların korunması, yalnızca eğitimden zorluk hesabı ve sonlu eğitim bütçesi makalenin deney düzeninden ayrılır. Bunlar makale düzeyi sonuçların birebir yeniden üretildiği iddiasını engeller. Promptsuz kontrol yeni bir MLP başlığı ve zorluk dahil tam hedef etkileşimi kullanır; KPT ailesi makaledeki e,c,r şablonunu ve ön eğitimli başlıkları kullanır. Dolayısıyla ana üçlü bütün yöntem paketlerini karşılaştırır; tek başına prompt eklemenin nedensel etkisini ayırmaz. Eski ve yeni sürüm de birden fazla bileşende değiştiğinden farklar tek bir düzeltmeye atfedilmez. Gerekçeli denetim [yöntem uyumu belgesindedir](makale_yontem_uyumu.md).", "",
        f"Toplam yeni deney süresi {results['total_seconds']/60:.2f} dakika; en yüksek ayrılan GPU belleği {results['peak_allocated_gpu_mb']:.1f} MiB. Sonuçlar outputs/main/paper_alignment_v2 altında, ham tahminlerden yeniden hesaplanan denetim analysis/integrity.json içindedir."]
    (ROOT / "docs/duzeltilmis_yontem_sonuclari.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"conditions": integrity["conditions"], "prototype_seed": selection["seed"], "summary": summary}), flush=True)
    subprocess.run([sys.executable,str(ROOT/"scripts/evaluate_attention_interpretation.py")],cwd=ROOT,check=True)
    block = read(RUN / "interpretations/results.json")
    extra = ["", "## Aynı hedef zaman bloğu yorumu (ayrı tanısal kontrol)", "",
        "Şablon sırası korunur; hedefin bilinen veya MASK tokenları aynı zaman adımında iki yönlü erişir. Geçmiş konumları sonraki etkileşimleri göremez. Bu, doğrulanmış yazar maskesi değildir. Aynı kilitli ağırlıklarda altı ek tahmin değerlendirmesi yapıldı; yeni eğitim veya bu maske için dönem seçimi yapılmadı. On iki ana eğitim sonucu yukarıda katı token yorumu olarak korunur.", "",
        "|Yöntem|Blok AUC ortalama|Blok doğruluk|Blok BCE|", "|---|---:|---:|---:|"]
    for method, metrics in block["summary"].items():
        extra.append(f"|{method}|{metrics['auc']['mean']:.4f}|{100*metrics['accuracy']['mean']:.2f}%|{metrics['bce']['mean']:.4f}|")
    chosen = block["prototype_selection"]
    extra += ["", f"Çalışan prototip için blok politikası testten önce kaydedildi; KMPT başlangıcı yalnızca bu politikadaki doğrulama BCE değeriyle seçildi: {chosen['seed']}. Seçilen modelde pseudo token değişikliğinin en büyük olasılık etkisi {block['selected_model_prompt_probability_delta']:.6g}. Güncel seçim interpretations/prototype_selection.json dosyasındadır; yukarıdaki analysis seçimi katı yorumun referans kaydıdır."]
    with (ROOT / "docs/duzeltilmis_yontem_sonuclari.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(extra)+"\n")


if __name__ == "__main__":
    main()
