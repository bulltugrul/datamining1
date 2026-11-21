"""Ana deney sonuçlarını açıklayan dördüncü çalışma defterini oluştur."""

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main():
    md = nbf.v4.new_markdown_cell
    code = lambda text: nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Knowledge Tracing (proje)",
                              "language": "python", "name": "kt-project"}
    nb.cells = [
        md("# Üç Başlangıçlı Ana Deneyler\n\nBu defter tam eğitim ve doğrulama verisiyle yürütülen promptsuz uyarlama, KPT ve KMPT deneylerinin bağımsız test sonuçlarını inceler. Dokuz koşul tamamlanmadan sonuç analizi çalıştırılmaz. Defter model eğitimi başlatmaz ve test sonuçlarına göre ayar değiştirmez."),
        code('''
            from pathlib import Path
            import json
            import sys
            import inspect
            import subprocess
            import pandas as pd
            from IPython.display import display, Image, Markdown, Code

            ROOT = Path.cwd().resolve()
            if not (ROOT / "src" / "kt_project").exists():
                ROOT = ROOT.parent
            assert (ROOT / "src" / "kt_project").exists()
            if str(ROOT / "src") not in sys.path:
                sys.path.insert(0, str(ROOT / "src"))
            RUN = ROOT / "outputs" / "main" / "three_seed_v1"
            status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
            assert status["status"] == "complete", "Ana deneylerin tamamlanması gerekiyor."
        '''),
        md("## Sonuçları ham tahminlerden denetleme\n\nAnaliz, üç yöntem ve üç başlangıcın eksiksiz olduğunu kontrol eder. Bütün koşulların aynı test hedeflerini kullandığı ve kaydedilmiş tahminlerin gerçek etiketlerle eşleştiği doğrulanır. AUC, doğruluk ve BCE değerleri kaydedilmiş olasılıklardan yeniden hesaplanır."),
        code('''
            analyzed = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_main.py")],
                                      cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
            results = json.loads((RUN / "results.json").read_text(encoding="utf-8"))
            summary = json.loads((RUN / "analysis" / "summary.json").read_text(encoding="utf-8"))
            integrity = json.loads((RUN / "analysis" / "integrity.json").read_text(encoding="utf-8"))
            assert integrity["all_checks_passed"] and integrity["complete_method_seed_pairs"] == 9
            display(pd.DataFrame([{
                "Tamamlanan karşılaştırma": integrity["complete_method_seed_pairs"],
                "Test öğrencisi": integrity["test_students"],
                "Ortak test hedefi": integrity["test_targets"],
                "Eğitimde görülmeyen soru hedefi": integrity["unseen_question_test_targets"],
                "Kaynak içerik özetleri doğrulandı": integrity["implementation_hashes_verified"],
            }]))
        '''),
        md("## Gerçek deney ayarları\n\nAyarlar her koşul için aynı kalır. Her başlangıç içinde LKM bir kez ön eğitilir ve üç yönteme kopyalanır. Öğrenci ayrımı başlangıçlar arasında değişmez. Testte her öğrencinin ilk etkileşimi geçmiş başlangıcı olarak kullanılır; diğer cevaplar birer tahmin hedefidir."),
        code('''
            settings = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
            display(pd.DataFrame([{
                "Başlangıçlar": str(settings["seeds"]),
                "Gizli boyut": settings["model"]["hidden_size"],
                "LKM katmanı": settings["model"]["layers"],
                "En uzun geçmiş": settings["model"]["max_history"],
                "Toplu örnek sayısı": settings["batch_size"],
                "Ön eğitim dönemi": settings["pretraining_epochs"],
                "En fazla uyarlama dönemi": settings["finetuning_epochs"],
                "Erken durdurma sabrı": settings["patience"],
                "Hassasiyet": results["precision"],
            }]))
            for seed in settings["seeds"]:
                assert len({row["initial_lkm_sha256"] for row in results["runs"] if row["seed"] == seed}) == 1
            print("Her başlangıçta üç yöntemin LKM başlangıcı aynı.")
        '''),
        md("## Üç başlangıcın ortalaması ve değişkenliği\n\nStandart sapma üç bağımsız başlangıç üzerinden, örnek standart sapması olarak hesaplanır. Bu değer aynı öğrenci ayrımındaki optimizasyon değişkenliğidir; öğrenci evreninin güven aralığı değildir. Ölçüler etkileşim ağırlıklıdır."),
        code('''
            names = {"baseline": "Promptsuz uyarlama", "kpt": "KPT", "kmpt": "KMPT"}
            rows = []
            for method in ("baseline", "kpt", "kmpt"):
                metrics = summary[method]["test"]
                rows.append({
                    "Yöntem": names[method],
                    "Test AUC ortalama ± SS": f"{metrics['auc']['mean']:.4f} ± {metrics['auc']['sample_sd']:.4f}",
                    "Test doğruluğu yüzde ortalama ± SS": f"{100*metrics['accuracy']['mean']:.2f} ± {100*metrics['accuracy']['sample_sd']:.2f}",
                    "Test BCE ortalama ± SS": f"{metrics['bce']['mean']:.4f} ± {metrics['bce']['sample_sd']:.4f}",
                })
            display(pd.DataFrame(rows))
            display(Image(filename=str(ROOT / "outputs" / "figures" / "main_test_comparison.png")))
        '''),
        md("## Ortalamanın nasıl hesaplandığı\n\nAna tablo eksik koşullardan üretilemez. Aynı başlangıçtaki yöntem farkları ayrıca hesaplanır. Üç ölçü için farklı dönemler seçilmez; doğrulama BCE ile seçilen dönem bütün ölçülerde kullanılır."),
        code('''
            from kt_project.experiment_results import aggregate_runs
            display(Code(inspect.getsource(aggregate_runs), language="python"))
        '''),
        md("## Her başlangıcın ayrı sonucu\n\nTek bir iyi koşulu seçmek yerine bütün koşullar gösterilir. Dönem seçiminde bağımsız test ölçüleri kullanılmaz."),
        code('''
            display(pd.DataFrame([{
                "Başlangıç": row["seed"], "Yöntem": names[row["method"]],
                "Seçilen dönem": row["best_epoch"],
                "Doğrulama BCE": row["validation"]["bce"],
                "Test AUC": row["test"]["auc"], "Test BCE": row["test"]["bce"],
                "Test doğruluğu": row["test"]["accuracy"],
            } for row in results["runs"]]).round(4))
            display(Image(filename=str(ROOT / "outputs" / "figures" / "main_validation_curves.png")))
        '''),
        md("## Test ROC eğrileri\n\nHer yöntemin üç başlangıcı aynı test hedeflerinde değerlendirilir. Eğriler, bu koşullarda eşik değiştikçe doğru ve yanlış pozitif oranlarının nasıl değiştiğini gösterir."),
        code('''
            display(Image(filename=str(ROOT / "outputs" / "figures" / "main_roc_curves.png")))
        '''),
        md("## Örnek oturum modelinin seçimi\n\nÖrnek oturum için yöntem önceden KMPT olarak belirlenmiştir. Kullanılacak başlangıç, KMPT'nin üç doğrulama BCE değeri arasından seçilir. Bu seçim, hangi yöntemin testte daha iyi olduğuna ilişkin karşılaştırmadan ayrıdır."),
        code('''
            selection = json.loads((RUN / "analysis" / "prototype_selection.json").read_text(encoding="utf-8"))
            assert selection["test_metrics_used_for_selection"] is False
            display(pd.DataFrame([selection]))
        '''),
        md("## Projenin otomatik güvenilirlik kontrolleri\n\nVeri ayrımı, modelin bilgi sınırı, çalışma defteri başlatıcısı ve sonuç analizinin kontrolleri birlikte çalıştırılır. Bu kontroller modelleri yeniden eğitmez ve bağımsız test sonuçlarına göre ayar değiştirmez."),
        code('''
            checked = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                     cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            transcript = checked.stdout + checked.stderr
            print(transcript)
            assert checked.returncode == 0, "Otomatik güvenilirlik kontrolü başarısız."
            (ROOT / "outputs" / "validation" / "main_test_output.txt").write_text(transcript, encoding="utf-8")
        '''),
        md("## Sonuçların yorumu ve uygulamanın sınırları\n\nBu karşılaştırma model uyarlama yöntemlerinin toplam etkisini ölçer. Saf PEFT deneyi veya yalnızca soft promptlara ait bir ablasyon değildir. Model kapasitesi ve veri işleme ayrıntıları makalenin birebir tekrarından farklıdır."),
        code('''
            display(Markdown((ROOT / "docs" / "ana_deney_sonuclari.md").read_text(encoding="utf-8")))
        '''),
    ]
    path = ROOT / "notebooks/04_Ana_Deneyler_ve_Karsilastirma.ipynb"
    nbf.validate(nb)
    nbf.write(nb, path)
    print(path.name)


if __name__ == "__main__":
    main()
