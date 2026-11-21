"""Öğrenci takibi defterlerini açıklamalar ve yeniden çalışabilir hücrelerle oluştur."""

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


BOOTSTRAP = '''
from pathlib import Path
import inspect
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Image, Code

ROOT = Path.cwd().resolve()
if not (ROOT / "src/kt_project").exists():
    ROOT = ROOT.parent
assert (ROOT / "src/kt_project").exists(), "Defteri proje klasöründen açın."
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime/cuda"))
os.environ.setdefault("PYTHONUTF8", "1")

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))

def run_script(name):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / name)],
                            cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
    result.check_returncode()

if not (ROOT / "outputs/patterns/behavior_model.joblib").exists():
    run_script("fit_behavior_patterns.py")
if not (ROOT / "outputs/replay/session.json").exists():
    run_script("run_student_replay.py")
metadata = read_json("outputs/patterns/metadata.json")
session = read_json("outputs/replay/session.json")
print("Proje klasörü:", ROOT)
'''


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Knowledge Tracing (proje)", "language": "python", "name": "kt-project"},
        "language_info": {"name": "python", "version": "3.13.2"},
    })
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / "notebooks" / name)
    print(name)


def main():
    write("05_Ogrenci_Gostergeleri_ve_Davranis_Gruplari.ipynb", [
        md('''
        # Öğrenci göstergeleri ve davranış grupları

        Bu defter, sabit referans sorularından beceri tahmini üretir ve eğitim
        geçmişlerinden öğrenilmiş davranış gruplarını açıklar. Önceki ana deney
        ağırlıkları kullanılır. İlk açılışta kayıtlı gruplar yoksa yalnızca bu aşama
        hesaplanır; ana model eğitimi yeniden başlatılmaz.
        '''),
        code(BOOTSTRAP),
        md('''
        ## Birbirinden farklı üç ölçü

        **Beceri tahmini:** eğitim sorularından önceden seçilmiş sabit referans
        kümesindeki doğru cevap olasılıklarının ortalaması × 100. Model en fazla
        yapılandırmada belirlenen uzunluktaki gözlenmiş geçmişi kullanır. **Gözlenen oran:** son 20 etkileşimde ilgili
        beceriye ait doğru cevap oranı; mutlaka cevap sayısıyla sunulur.
        **Model doğruluğu:** ayrı test öğrencilerinde ölçülen sınıflandırma
        başarısıdır. İlk sürüm dördüncü, yöntem denetiminden sonraki sonuçlar sekizinci defterdedir.

        Prototip için önceden belirlenen KMPT yöntemi içinde başlangıç yalnızca
        doğrulama BCE değerine göre seçildi. Burada kullanılan KMPT,
        testte en iyi yöntem olduğu iddiasıyla seçilmez; güncel karşılaştırma ayrı gösterilir.
        Güncel prototip aynı hedef zaman bloğu dikkat yorumunu kullanır. Katı token
        yorumu ve iki yorum arasındaki fark sekizinci defterde ayrı incelenir.
        '''),
        code('''
        import torch
        from kt_project.sequences import InteractionStore
        from kt_project.tracking import FrozenTracer, ReferenceBank, snapshot_for

        torch.set_num_threads(4)
        store = InteractionStore(ROOT / "data/processed/interactions.npz")
        bank = ReferenceBank(ROOT / "data/processed/references.json")
        tracer = FrozenTracer(ROOT, store)
        print("Etkin model seçimi:", tracer.selection)
        print("En uzun gözlenmiş geçmiş:", tracer.config.max_history)
        student = session["encoded_student"]
        start, _ = store.spans[student]
        snapshot = snapshot_for(store, student, start + session["initial"]["history_count"])
        skill = session["initial"]["skill_scores"][0]["skill"]
        references = bank.for_skill(skill)
        probabilities = tracer.predict_requests([(snapshot, item) for item in references])
        reference_percent = 100 * float(probabilities.mean())
        assert np.isclose(reference_percent,
                          session["initial"]["skill_scores"][0]["predicted_success_percent"], atol=0.1)
        assert bank.source_sha256 == metadata["reference_bank_sha256"]
        assert tracer.provenance["checkpoint_sha256"] == metadata["checkpoint_sha256"]
        assert tracer.provenance["inference_policy"] == metadata["inference_policy"]
        display(pd.DataFrame({"Referans soru kimliği": [r["question"] for r in references],
                              "Tahmini doğru cevap olasılığı": probabilities}).head(8))
        print(f"{len(references)} sabit sorunun ortalaması: %{reference_percent:.2f}")
        display(pd.DataFrame(session["initial"]["skill_scores"]))
        '''),
        md('''
        ## Gruplamaya giren yakın dönem özellikleri

        Her pencere dört değer içerir: doğru cevap oranı, geçmişte en sık görülen
        en fazla üç becerinin referans tahmin ortalaması, pozitif yanıt sürelerinin
        medyanına `log1p` dönüşümü ve cevap başına ortalama ipucu sayısı.
        Aktif beceri ortalaması bağlamsal bir gruplama girdisidir.

        Süre ve ipucu sınırları eğitimden öğrenilir. Sıfır veya geçersiz süreler
        eksik kabul edilir; eğitim medyanları ve eğitim ölçeği kullanılır.
        Aşağıda gerçekten kullanılan özellik çıkarma kodu gösterilir.
        '''),
        code('''
        from kt_project.patterns import behavior_vector, FEATURE_NAMES
        display(Code(inspect.getsource(behavior_vector), language="python"))
        display(pd.Series(metadata["limits"], name="Eğitimden öğrenilen sınırlar"))
        '''),
        md('''
        ## Grup sayısının seçimi ve profiller

        Öğrenci başına en fazla dört eğitim geçmişi kullanılır. En az beş cevap
        gerekir. İki ile altı grup arasında, her grubun en az yüzde 5 pencere
        içermesi koşuluyla en yüksek örneklenmiş siluet değeri seçilir.
        KMeans++ başlangıcı, 10 başlangıç ve sabit 2026 rastgele başlangıcı kullanılır.
        Siluet değeri sınıflandırma doğruluğu veya öğrenme kazanımı yüzdesi değildir.
        '''),
        code('''
        print("Eğitim öğrencisi:", metadata["training_students"])
        print("Eğitim geçmişi penceresi:", metadata["training_snapshots"])
        display(pd.DataFrame(metadata["candidates"]))
        print("Seçilen grup:", metadata["selected_clusters"],
              "Siluet:", round(metadata["selected_silhouette"], 4))
        display(pd.DataFrame(metadata["profiles"]))
        display(Image(filename=str(ROOT / "outputs/figures/pattern_selection_and_groups.png")))
        display(Image(filename=str(ROOT / "outputs/figures/pattern_profiles.png")))
        '''),
        md('''
        Aynı öğrencinin farklı zamanlardaki pencereleri farklı gruplara girebilir;
        profillerdeki öğrenci sayıları toplanmaz. Bunlar kalıcı öğrenme tarzları
        değildir. Referans özelliği eğitim geçmişlerinde aynı sabit modelden
        üretildiği için burada ayrıca bağımsız gruplama başarısı iddia edilmez.

        ## Güncel geçmişi öğrenilmiş gruba atama

        Aşağıdaki işlem model ve grupları yeniden eğitmeden ilk oturum durumunu
        hesaplar. Merkeze uzaklık güven olasılığı olarak yorumlanmaz.
        '''),
        code('''
        import joblib
        patterns = joblib.load(ROOT / "outputs/patterns/behavior_model.joblib")
        active = session["initial"]["active_pattern_skills"]
        scores = tracer.reference_scores([snapshot], [active], bank)[0]
        feature = behavior_vector(store, snapshot, scores, patterns.limits, patterns.window_size)
        assignment = patterns.predict(feature)
        assert assignment["cluster"] == session["initial"]["behavior"]["cluster"]
        display(pd.Series(feature, index=FEATURE_NAMES, name="Güncel özellikler"))
        print(assignment)
        '''),
        md('''
        **Sonraki defter:** gerçek kayıtları sırayla işleyerek göstergelerin,
        grubun ve çalışma önerisinin cevaplardan sonra nasıl değiştiğini gösterir.
        Ayrıntılı yöntem `docs/ogrenci_takibi_ve_oruntuler.md` dosyasındadır.
        '''),
    ])

    write("06_Ornek_Oturum_ve_Calisma_Onerileri.ipynb", [
        md('''
        # Örnek öğrenme oturumu ve çalışma önerileri

        Doğrulama kümesinden, yeterli geçmiş uzunluğu ve sabit rastgele başlangıçla
        seçilmiş bir öğrencinin ilk beş cevabı başlangıçtır. Sonraki 20 gerçek
        kayıt yeniden oynatılır. Öğrenci, sonuç eğrisine göre seçilmemiştir.
        Tahmin önce üretilir; gerçek cevap daha sonra geçmişe eklenir.
        '''),
        code(BOOTSTRAP),
        md('''
        ## Cevaptan önce tahmin ve cevaptan sonra güncelleme

        Model ağırlıkları, referans sorular ve öğrenilmiş gruplar sabittir.
        Her cevap sonrası öğrenci geçmişi yenilenir. Aşağıdaki fonksiyon gerçek
        akışı gösterir; kayıtlı çıktı bu fonksiyonla üretilmiştir.
        '''),
        code('''
        from kt_project.replay import run_replay
        display(Code(inspect.getsource(run_replay), language="python"))
        states = [session["initial"]] + [event["after"] for event in session["events"]]
        events = pd.DataFrame([{
            "Adım": event["step"], "Mevcut soru kimliği": event["original_problem_id"],
            "Cevaptan önce tahmin %": 100*event["predicted_before_response"],
            "Gerçek cevap (1 doğru)": event["observed_correct"],
            "Son 20 cevapta doğru %": 100*event["after"]["observed_recent_correct_ratio"],
            "Cevaptan sonraki grup": event["after"]["behavior"]["cluster"],
        } for event in session["events"]])
        display(events.round(2))
        groups = [state["behavior"]["cluster"] for state in states]
        print("Grup geçiş sayısı:", sum(a != b for a, b in zip(groups, groups[1:])))
        display(Image(filename=str(ROOT / "outputs/figures/replay_predictions_and_groups.png")))
        '''),
        md('''
        ## Beceri göstergelerinin cevap adımlarındaki değişimi

        Mavi çizgi sabit referans sorularındaki tahmindir; kesikli çizgi yakın
        dönem gözlenen orandır. Yalnızca bir doğru cevaba dayanan yüzde 100 oran
        kesin öğrenme yüzdesi olarak yorumlanmaz. Çizgiler ayrık cevap adımlarını
        birleştirir; aralarda sürekli öğrenme ölçümü yapılmamıştır.
        '''),
        code('''
        display(Image(filename=str(ROOT / "outputs/figures/replay_skill_progress.png")))
        display(pd.DataFrame(states[-1]["skill_scores"]).rename(columns={
            "name": "Beceri", "predicted_success_percent": "Referans tahmini %",
            "observed_correct_percent": "Gözlenen doğru %", "recent_skill_answers": "Yakın dönem cevap sayısı",
            "reference_question_count": "Sabit referans soru sayısı"}).round(2))
        '''),
        md('''
        ## Gerekçeli mevcut soru önerileri

        Yakın geçmişte aktif becerilerden referans tahmini en düşük olan seçilir.
        Hedef doğru cevap olasılığı normalde yüzde 65, son üç cevap yanlışsa
        yüzde 80, son üç cevap doğruysa yüzde 55'tir. Eğitim soruları arasından
        bu hedefe yakın ve mümkünse önceden cevaplanmamış bir soru önerilir.
        Eşikler açık kurallardır; öğrenilmiş öneri politikası değildir.

        Kaynakta soru metni bulunmadığından gerçek soru kimliği gösterilir.
        Yeniden oynatımda izleyen kayıt, önerilen soru olmak zorunda değildir;
        bu nedenle önerinin öğrenme etkisi burada ölçülmez.
        '''),
        code('''
        from kt_project.recommendations import challenge_probability
        display(Code(inspect.getsource(challenge_probability), language="python"))
        recommendations = pd.DataFrame([{
            "Adım": i, "Önerilen mevcut soru": state["recommendation"].get("original_problem_id"),
            "Beceri": state["recommendation"].get("skill_name"),
            "Hedef doğru %": 100*state["recommendation"].get("target_probability", float("nan")),
            "Soru tahmini %": 100*state["recommendation"].get("predicted_probability", float("nan")),
            "Tekrar sorusu": state["recommendation"].get("repeat_question"),
        } for i, state in enumerate(states)])
        display(recommendations.round(2))
        display(Image(filename=str(ROOT / "outputs/figures/replay_recommendation_targets.png")))
        '''),
        md('''
        ## Sunum için bir adımı inceleme

        `step_number` değerini 0–20 arasında değiştirip bu hücreyi çalıştırın.
        Sıfır başlangıç durumudur. Her seçimde aynı kayıtlı oturumun o andaki
        göstergeleri ve öneri gerekçesi gösterilir.
        '''),
        code('''
        step_number = 10
        if not 0 <= step_number < len(states):
            raise ValueError(f"Adım 0 ile {len(states)-1} arasında olmalı.")
        selected = states[step_number]
        print("Adım:", step_number, "Geçmiş cevap sayısı:", selected["history_count"])
        display(pd.DataFrame(selected["skill_scores"]).round(2))
        print("Davranış grubu:", selected["behavior"]["cluster"], selected["behavior"]["description"])
        print("Öneri:", selected["recommendation"]["reason"])
        '''),
        md('''
        ## Zaman bilgisinin anlamı ve güvenilirlik denetimi

        Kaynakta bu kayıtların gerçek takvim zamanları bulunmuyor. Sıralama gerçek
        kaynak sıra kimliğiyle korunur. Gösterim zamanı programın çalıştığı ana
        aittir; öğrencinin geçmişte soruyu cevapladığı tarih değildir. Bu oturum
        zamana yayılmış gerçek öğrenme kazanımı deneyi olarak sunulmaz.
        '''),
        code('''
        display(pd.DataFrame([{
            "Adım": event["step"], "Kaynak sıra kimliği": event["source_order_id"],
            "Kaynak takvim zamanı": event["source_timestamp"],
            "Program gösterim zamanı": event["display_timestamp"],
        } for event in session["events"]]).iloc[[0, -1]])
        display(pd.Series(session["integrity"], name="Oturum kontrolleri"))
        run_script("audit_student_tracking.py")
        display(pd.Series(read_json("outputs/validation/student_tracking_integrity.json")))
        '''),
        code('''
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                                 "-p", "test_student_tracking.py", "-v"],
                                cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        test_output = result.stdout + result.stderr
        (ROOT / "outputs/validation/tracking_test_output.txt").write_text(test_output, encoding="utf-8")
        print(test_output)
        result.check_returncode()
        '''),
    ])


if __name__ == "__main__":
    main()
