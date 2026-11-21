"""Model açıklamasını ve gerçekten çalışmış küçük deneme defterini üret."""

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/pilot/lkm_128_device_batches"


def figures():
    pretraining = json.loads((RUN / "seed_17/pretraining_history.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    titles = [("question_loss", "RTM soru kaybı"), ("response_loss", "RTM cevap kaybı"),
              ("order_loss", "FBO sıralama kaybı")]
    for ax, (key, title) in zip(axes, titles):
        ax.plot([row["epoch"] for row in pretraining], [row[key] for row in pretraining],
                marker="o", color="#2463a6")
        ax.set_title(title)
        ax.set_xlabel("Dönem")
        ax.set_ylabel("Ortalama kayıp")
        ax.set_xticks([row["epoch"] for row in pretraining])
        ax.grid(alpha=.2)
    fig.suptitle("Küçük denemede LKM ön eğitimi", fontsize=14)
    out = ROOT / "outputs/figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "pilot_pretraining.png", dpi=160)
    plt.close(fig)
    results = json.loads((RUN / "results.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for record, color, label in zip(results["runs"], ["#52647c", "#2463a6", "#c56b14"],
                                     ["Promptsuz uyarlama", "KPT", "KMPT"]):
        history = json.loads((RUN / f"seed_17/{record['method']}_history.json").read_text())
        ax.plot([row["epoch"] for row in history], [row["validation"]["bce"] for row in history],
                marker="o", color=color, label=label)
        ax.scatter([record["best_epoch"]], [record["validation"]["bce"]], s=120,
                   facecolors="none", edgecolors=color, linewidths=2)
    ax.set(title="Küçük denemede doğrulama kaybı", xlabel="Dönem", ylabel="Cevap tahmini BCE")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out / "pilot_validation_bce.png", dpi=160)
    plt.close(fig)


def notebook():
    md = nbf.v4.new_markdown_cell
    code = lambda text: nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Knowledge Tracing (proje)",
                              "language": "python", "name": "kt-project"}
    nb.cells = [
        md("# LKM Modeli ve Küçük Eğitim Denemesi\n\nBu defter LKM, RTM/FBO ön eğitimi, PromEnc, KPT ve KMPT uygulamasını ve küçük denemenin gerçek çıktılarını açıklar. Deneme tek başlangıç ve sınırlı eğitim/doğrulama örneği kullanır. Bağımsız test verisi değerlendirilmez; bu sonuçlar ana bilimsel karşılaştırma değildir."),
        code('''
            from pathlib import Path
            import json
            import sys
            import inspect
            import subprocess
            import re
            import pandas as pd
            from IPython.display import display, Image, Markdown, Code

            ROOT = Path.cwd().resolve()
            if not (ROOT / "src" / "kt_project").exists():
                ROOT = ROOT.parent
            assert (ROOT / "src" / "kt_project").exists()
            if str(ROOT / "src") not in sys.path:
                sys.path.insert(0, str(ROOT / "src"))
            RUN = ROOT / "outputs" / "pilot" / "lkm_128_device_batches"
        '''),
        md("## Küçük denemeyi çalıştırma veya kaydını açma\n\nTamamlanmış deneme varsa ağırlıklar ve çıktılar yeniden kullanılır. Yeni kurulumda yalnızca pilot ayarlarıyla küçük deneme çalışır; ana deney otomatik başlatılmaz."),
        code('''
            if not (RUN / "results.json").exists():
                completed = subprocess.run([
                    sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
                    "--config", "configs/pilot.json", "--output", str(RUN)
                ], cwd=ROOT, check=True)
            status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
            assert status["status"] == "complete"
            results = json.loads((RUN / "results.json").read_text(encoding="utf-8"))
            settings = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
            assert results["test_evaluated"] is False
            assert all(row["test"] is None for row in results["runs"])
            display(pd.DataFrame([{
                "Aygıt": results["gpu"], "Hassasiyet": results["precision"],
                "Eğitim öğrencisi üst sınırı": settings["train_student_limit"],
                "Uyarlama eğitim hedefi": results["runs"][0]["training_target_count"],
                "Doğrulama hedefi": results["runs"][0]["validation_target_count"],
                "En yüksek ayrılan GPU belleği MB": round(results["peak_allocated_gpu_mb"], 1),
            }]))
        '''),
        md("## Model yapısı\n\nLKM iki katmanlı, dört dikkat başlıklı ve 128 boyutlu bir Transformer kodlayıcısıdır. Soru, beceri, cevap, zorluk ve konum temsillerini kullanır. PromEnc bir katman ve sekiz öğrenilebilir prompt içerir. Üç yöntemde de LKM uyarlanır; deney saf PEFT karşılaştırması değildir."),
        code('''
            from kt_project.models import LargeKnowledgeModel, KnowledgeTracer
            display(pd.DataFrame([{
                "Yöntem": row["method"], "Toplam parametre": row["parameters"]["total"],
                "Eğitilen parametre": row["parameters"]["trainable"],
            } for row in results["runs"]]))
            display(Markdown((ROOT / "docs" / "model_yontemi.md").read_text(encoding="utf-8")))
        '''),
        md("## RTM ve FBO kayıplarını hesaplayan kod\n\nMaskeli soru ve cevaplar ayrı başlıklarla geri bulunur. CLS konumu iki parçanın doğru ardışıklığını tahmin eder. Kayıplar birlikte optimize edilir; her birinin değeri ayrı kaydedilir."),
        code('''
            display(Code(inspect.getsource(LargeKnowledgeModel.pretraining_loss), language="python"))
            history = json.loads((RUN / "seed_17" / "pretraining_history.json").read_text(encoding="utf-8"))
            display(pd.DataFrame(history))
            display(Image(filename=str(ROOT / "outputs" / "figures" / "pilot_pretraining.png")))
        '''),
        md("## Hedef cevabın gizlenmesi ve promptların görünürlüğü\n\nNormal tahmin yolunda label, hedef cevabın embedding girdisine alınmaz. Soft promptlar maskeli hedef bileşenden önce yerleştirilir; böylece nedensel LKM bunları görebilir. Yardımcı soru tahmini yalnızca KMPT eğitimi sırasında gerçek hedef cevabını kullanabilir."),
        code('''
            display(Code(inspect.getsource(KnowledgeTracer.encode_prediction), language="python"))
        '''),
        md("## Ortak başlangıç ve doğrulama sonuçları\n\nÜç yöntem aynı LKM ağırlığından başlar. En iyi dönem doğrulama BCE değerine göre seçilir; AUC ve doğruluk o aynı dönemin değerleridir. Birden fazla bileşen değiştiği için farklar yalnızca soft promptların etkisi olarak yorumlanmaz."),
        code('''
            assert len({row["initial_lkm_sha256"] for row in results["runs"]}) == 1
            display(pd.DataFrame([{
                "Yöntem": row["method"], "BCE ile seçilen dönem": row["best_epoch"],
                "Başlangıç doğrulama BCE": row["initial_validation"]["bce"],
                "Seçilen doğrulama BCE": row["validation"]["bce"],
                "Seçilen doğrulama AUC": row["validation"]["auc"],
                "Seçilen doğrulama doğruluğu": row["validation"]["accuracy"],
            } for row in results["runs"]]).round(4))
            display(Image(filename=str(ROOT / "outputs" / "figures" / "pilot_validation_bce.png")))
        '''),
        md("## Kayıtlı ağırlıkları yeniden yükleyerek denetleme\n\nAğırlıklar yeniden yüklenir ve doğrulama ölçülerinin yeniden üretildiği kontrol edilir. Hedef etiket değiştirildiğinde tahminin değişmediği, KPT ve KMPT promptlarının CUDA üzerinde gradyan aldığı doğrulanır."),
        code('''
            audited = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_pilot.py")],
                                     cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
            audit = json.loads(audited.stdout)
            assert audit["common_pretrained_state_verified"] and audit["test_not_evaluated"]
            display(pd.DataFrame(audit["checks"]))
        '''),
        md("## Veri ve model güvenilirliği testleri\n\nTestler öğrenci ayrımı, zorluğun eğitimden hesaplanması, hedef cevabın gizlenmesi, nedensellik, dolgu bağımsızlığı ve promptların gerçekten güncellenmesini denetler. Geçerli test çıktısı kaydedilir."),
        code('''
            checked = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                     cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            transcript = checked.stdout + checked.stderr
            print(transcript)
            assert checked.returncode == 0, "Güvenilirlik kontrolü başarısız."
            (ROOT / "outputs" / "validation" / "model_test_output.txt").write_text(transcript, encoding="utf-8")
            count = int(re.search(r"Ran (\\d+) tests", transcript).group(1))
            print("Başarılı kontrol sayısı:", count)
        '''),
        md("## Ana deney hazırlığı\n\nAna deney ayarları configs/main.json dosyasında ayrı tutulur. Üç başlangıç, tam eğitim ve doğrulama öğrencileri, ortak ön eğitim ve ayrılmış test değerlendirmesi kullanılacaktır. Küçük denemedeki yöntem sıralaması veya sayılar nihai sonuç olarak aktarılmaz. Çalışan modelin hazırlanması ile ana deneyin tamamlanması ayrı aşamalardır."),
    ]
    nbf.validate(nb)
    output = ROOT / "notebooks/03_LKM_Modeli_ve_Kucuk_Deneme.ipynb"
    nbf.write(nb, output)
    print(output.name)


if __name__ == "__main__":
    figures()
    notebook()
