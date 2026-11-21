"""Makale karşılaştırması ve düzeltilmiş eğitimlerin sonuç defteri."""

from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def main():
    notebook = nbf.v4.new_notebook(cells=[
        md('''
        # Makale yöntem denetimi ve düzeltilmiş deneyler

        Bu defter, makalenin Bölüm 3, Denklem 10–13 ve Algoritma 1 tariflerini
        uygulamayla karşılaştırır. Dördüncü defter önceki sürümün tarihsel
        sonuçlarını korur. Buradaki yeni eğitimler eski ağırlıkların üzerine yazılmaz.

        Çekirdek yöntem farkları düzeltildi. Model kapasitesi ve değerlendirme
        düzenindeki korunan farklar nedeniyle çalışma ölçekli yöntem uygulamasıdır;
        makalenin bütün deneylerinin birebir tekrarı olarak sunulmaz.
        '''),
        code('''
        from pathlib import Path
        import inspect
        import json
        import os
        import subprocess
        import sys
        import pandas as pd
        from IPython.display import display, Image, Code

        ROOT = Path.cwd().resolve()
        if not (ROOT / "src/kt_project").exists():
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT / "src"))
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
        os.environ.setdefault("PYTHONUTF8", "1")
        RUN = ROOT / "outputs/main/paper_alignment_v2"
        def read(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))
        def run_script(name):
            completed = subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT,
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(completed.stdout)
            if completed.returncode:
                print(completed.stderr)
            completed.check_returncode()
        status = read(RUN / "status.json")
        assert status["status"] == "complete", "Eğitim tamamlanmadan sonuç defteri çalıştırılmaz."
        results, config = read(RUN / "results.json"), read(RUN / "config.json")
        audit = read(ROOT / "outputs/method_audit/audit_matrix.json")
        display(pd.DataFrame(audit["matrix"]))
        '''),
        md('''
        ## Düzeltilen şablon ve dikkat

        Eğitimde iki modül de iki yönlü; tahminde ikisi de nedenseldir. Her hedef
        ayrı bir geçmiş önekiyle işlendiği için iki yönlü eğitimde hedefin gerçek
        cevabı girdi olmaz. KPT şablonu `e,c,MASK_r,T`; KMPT yardımcı şablonu
        `MASK_e,c,r,T` sırasındadır. Çıktı son tokendan değil ilgili maskeden okunur.
        Şablonda hedef soru zorluğu eklenmez; geçmiş etkileşimlerinde zorluk korunur.
        '''),
        code('''
        from kt_project.plkt_models import PaperPromptEncoder, PaperKnowledgeTracer, PaperLKM
        from kt_project.plkt_sequences import PaperPretrainingDataset
        display(Code(inspect.getsource(PaperPromptEncoder.forward), language="python"))
        display(Code(inspect.getsource(PaperKnowledgeTracer.encode_prediction), language="python"))
        '''),
        md('''
        ## Ön eğitim, ölçekleme ve parametre güncellemesi

        FBO negatifleri bütün yarım parça havuzundan, özgün doğru çift hariç
        seçilir. RTM seçilmiş konumda soru ve cevap için ortak 80/10/10 bozulması
        uygular. Maskeli konumların kayıpları ile FBO kaybı aynı dizi sayısına göre
        ölçeklenir. LKM BERT tarzı son normalizasyon kullanır.

        Algoritma 1 adım 24 LKM'nin uyarlanmasını tarif eder; açık bir dondurma
        zorunluluğu yoktur. LKM ve promptların birlikte eğitilmesi korunmuştur.
        '''),
        code('''
        display(Code(inspect.getsource(PaperLKM.pretraining_loss), language="python"))
        display(pd.Series(config["model"]))
        display(pd.Series(config["retained_differences"], name="Korunan deney farkları"))
        stopping = []
        for seed in config["seeds"]:
            stopping.append({"seed": seed, **read(RUN / f"seed_{seed}/pretraining_stopping.json")})
        display(pd.DataFrame(stopping))
        '''),
        md('''
        ## Makale tarifindeki dikkat gerilimi için ayrı kontrol

        Sondaki pseudo tokenlar, katı nedensellikte önceki maskeli cevabı
        etkileyemez. Bu, makalenin sırası ve Bölüm 3.5 birlikte okunduğunda çıkan
        sonuçtur; yazar kodunun incelendiği anlamına gelmez. Ayrı KMPT prefix
        kontrolü yalnızca şablon yerleşimini değiştirir. Eğitimde prompt gradyanı
        bulunması, aynı promptun tahminde erişilebilir olduğunu kanıtlamaz.
        '''),
        code('''
        display(Image(filename=str(ROOT / "outputs/figures/paper_prompt_visibility.png")))
        run_script("analyze_paper_alignment.py")
        diagnostic = read(RUN / "analysis/prompt_sensitivity.json")
        display(pd.DataFrame(diagnostic))
        '''),
        md('''
        ## Aynı hedeflerde üç başlangıçlı yeni sonuçlar

        Ana karşılaştırma promptsuz, KPT ve KMPT'dir. Prefix ayrı yerleşim
        kontrolüdür. Bütün koşullar aynı başlangıç içinde aynı yeni ön eğitimden
        başlar. Dönemler yalnızca doğrulama BCE değeriyle seçilir; tüm eğitimler
        bittikten sonra kontrol noktaları kilitlenir ve test hesaplanır.
        '''),
        code('''
        summary = read(RUN / "analysis/summary.json")
        rows = []
        for variant, metrics in summary.items():
            rows.append({"Koşul": variant,
                "AUC ortalama": metrics["auc"]["mean"], "AUC örnek SS": metrics["auc"]["sample_sd"],
                "Doğruluk ortalama %": 100*metrics["accuracy"]["mean"],
                "BCE ortalama": metrics["bce"]["mean"]})
        display(pd.DataFrame(rows).round(4))
        display(pd.DataFrame([{"seed": row["seed"], "variant": row["variant"],
            "best_epoch": row["best_epoch"], "validation_BCE": row["validation"]["bce"],
            "test_AUC": row["test"]["auc"], "test_accuracy": row["test"]["accuracy"]} for row in results["runs"]]))
        display(Image(filename=str(ROOT / "outputs/figures/paper_alignment_comparison.png")))
        display(Image(filename=str(ROOT / "outputs/figures/paper_alignment_validation.png")))
        '''),
        md('''
        Sabit ayrımın test sonuçları önceki sürümde zaten görülmüştür. Yeni deney,
        tamamen dokunulmamış yeni bir doğrulayıcı test olarak sunulmaz. Yöntem
        düzeltmeleri eski testte üstün sonuç elde etmek için seçilmemiştir. Eski
        ve yeni sürüm birden çok bileşende değiştiğinden başarı farkı tek bir
        düzeltmeye atfedilemez. Üç başlangıçtaki standart sapma güven aralığı değildir.

        ## Güncel öğrenci takibine bağlantı ve denetim

        Ek kontrolde aynı hedefin bilinen şablon tokenları tek zaman bloğu
        kabul edilir. Geçmiş sonraki olayları göremez; aynı hedef bloğunda
        iki yönlü erişim vardır. Bu, doğrulanmış yazar kodu değil açık bir maske
        yorumudur. Aynı ağırlıklarda altı ek tahmin değerlendirmesi yapılır;
        yeni eğitim veya bu politikaya özel dönem seçimi yapılmaz. Prototipte
        bu blok politikası testten önce belirlenmiş, KMPT başlangıcı yalnızca
        blok doğrulama BCE değeriyle seçilmiştir.
        Öğrenci göstergeleri, davranış grupları, yeniden oynatım
        ve küçük etkileşim önerileri yeni ağırlıklarla yeniden üretilir. Eski
        gösterim çıktıları arşivlenir. Beşinci, altıncı ve yedinci defter güncel
        ağırlıkları kullanır.
        '''),
        code('''
        interpretation = read(RUN / "interpretations/results.json")
        display(pd.DataFrame([{"method": row["method"], "seed": row["seed"],
            "block_AUC": row["test"]["auc"], "strict_AUC": row["strict_test"]["auc"]}
            for row in interpretation["runs"]]))
        print("Blok politikada prompt olasılık etkisi:", interpretation["selected_model_prompt_probability_delta"])
        selection = read(RUN / "interpretations/prototype_selection.json")
        display(pd.Series(selection))
        integrity = read(RUN / "analysis/integrity.json")
        display(pd.Series({key: value for key, value in integrity.items() if key != "checks"}))
        if (ROOT / "configs/active_model.json").exists():
            active = read(ROOT / "configs/active_model.json")
            assert active["version"] == "paper_alignment_v2"
            replay = read(ROOT / "outputs/replay/session.json")
            patterns = read(ROOT / "outputs/patterns/metadata.json")
            assert replay["model_selection"]["checkpoint_sha256"] == selection["checkpoint_sha256"]
            assert patterns["checkpoint_sha256"] == selection["checkpoint_sha256"]
            assert patterns["inference_policy"] == selection["inference_policy"]
            print("Öğrenci takibi ve örüntü modeli güncel ağırlıklarla eşleşiyor.")
        '''),
        code('''
        completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        output = completed.stdout+completed.stderr
        (ROOT / "outputs/validation/method_alignment_test_output.txt").write_text(output, encoding="utf-8")
        print(output)
        completed.check_returncode()
        '''),
        md('''
        Kaynak: kullanıcı tarafından sağlanan Zhu, Qiu ve Zhou makalesi,
        DOI 10.1007/s10618-026-01208-9. Özellikle sayfa 9–15 ve 18.
        Madde bazlı açıklama `docs/makale_yontem_uyumu.md`, yeni sayısal sonuçlar
        `docs/duzeltilmis_yontem_sonuclari.md` dosyalarındadır. Teslim raporu bu
        denetimden sonra hazırlanır.
        '''),
    ], metadata={"kernelspec": {"display_name": "Knowledge Tracing (proje)", "language": "python", "name": "kt-project"},
                 "language_info": {"name": "python", "version": "3.13.2"}})
    nbf.validate(notebook)
    name = "08_Makale_Yontem_Denetimi_ve_Duzeltilmis_Deneyler.ipynb"
    nbf.write(notebook, ROOT / "notebooks" / name)
    print(name)


if __name__ == "__main__":
    main()
