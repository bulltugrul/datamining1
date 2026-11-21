"""Üç giriş yolunu gerçekten çalıştıran ve küçük örnekleri açıklayan defter."""

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
        # Metin, ses ve görsel etkileşim örnekleri

        Öğrenci, çalışmak istediği beceriyi yazarak, kısa bir ses dosyasıyla veya
        yazı içeren görselle ifade eder. Üç giriş aynı beceri belirleme ve mevcut
        soru önerme akışına gider. İstek bir ölçme sorusuna cevap değildir;
        öğrencinin başarı geçmişine eklenmez.

        Bu defterde ses gerçekten çözümlenir, görseldeki yazı gerçekten okunur.
        Önceki 20 adımlık örnek oturumun sonundaki 25 cevap sabit bağlamdır.
        '''),
        code('''
        from pathlib import Path
        import inspect
        import json
        import os
        import subprocess
        import sys
        import pandas as pd
        from IPython.display import display, Audio, Image, Code

        ROOT = Path.cwd().resolve()
        if not (ROOT / "src/kt_project").exists():
            ROOT = ROOT.parent
        assert (ROOT / "src/kt_project").exists()
        sys.path.insert(0, str(ROOT / "src"))
        os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/huggingface"))
        os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime/cuda"))
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))

        def read_json(name):
            return json.loads((ROOT / name).read_text(encoding="utf-8"))

        def run_script(name):
            process = subprocess.run([sys.executable, str(ROOT / "scripts" / name)],
                                     cwd=ROOT, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace")
            print(process.stdout)
            if process.returncode:
                print(process.stderr)
            process.check_returncode()

        import torch
        from kt_project.intent import train_intent_model
        from kt_project.interactions import InteractionService
        from kt_project.media import SpeechReader, read_image
        torch.set_num_threads(4)
        if not (ROOT / "outputs/interactions/intent_model.joblib").exists():
            train_intent_model(ROOT)
        if not (ROOT / "data/models/whisper-base/model.safetensors").exists():
            raise FileNotFoundError("Önce scripts/download_interaction_model.py ile açık ses modelini indirin.")
        if not (ROOT / "outputs/interactions/examples.json").exists():
            run_script("run_interaction_examples.py")
        metrics = read_json("outputs/interactions/intent_metrics.json")
        examples = read_json("outputs/interactions/examples.json")
        service = InteractionService(ROOT)
        print("Öğrenci bağlamı:", service.snapshot.stop-service.snapshot.start, "mevcut cevap")
        display(pd.Series(service.tracer.selection, name="Etkin bilgi takibi modeli"))
        '''),
        md('''
        ## Beceri isteğinin öğrenilmesi ve kapsam sınırı

        Üç beceri: tam sayılarda toplama ve çıkarma, pozitif ondalık sayılarda
        toplama ve çıkarma, kesirlerde toplama ve çıkarma. Bunlar mevcut ASSIST
        becerileridir. Karakter TF IDF ve lojistik regresyon 48 yazılmış eğitim
        ifadesinden öğrenir. 20 doğrulama ifadesi kabul eşiği seçimine ayrılır.
        20 başka ifade küçük işlev kontrolüdür. Gerçek öğrenci beyanları değildir.

        Kapsam kuralları çoklu beceri, belirsiz konu ve desteklenmeyen işlemde
        açıklama ister. Düşük sınıflandırma skorunda da konu zorla seçilmez.
        Bu skor öğrenci başarı yüzdesi veya kalibre edilmiş güven değildir.
        '''),
        code('''
        print(service.intent.pipeline)
        display(pd.DataFrame(metrics["threshold_candidates"]))
        display(pd.Series({key: metrics[key] for key in [
            "training_records", "validation_records", "test_records", "threshold", "test_accuracy", "test_macro_f1"]}))
        display(pd.DataFrame(metrics["test_predictions"]))
        if not (ROOT / "outputs/validation/interaction_integrity.json").exists():
            run_script("audit_interaction_examples.py")
        if not (ROOT / "outputs/figures/interaction_intent_confusion.png").exists():
            run_script("write_interaction_report.py")
        display(Image(filename=str(ROOT / "outputs/figures/interaction_intent_confusion.png")))
        '''),
        md('''
        Küçük ayrılmış derlemde 20 örneğin 19'u beklenen etiketi aldı. Kapsam içi
        bir tam sayı isteği düşük skor nedeniyle açıklama isteme sonucuna gitti;
        hatayı koruyoruz. Bu yazılmış derlemdeki oran, gerçek öğrencilerdeki veya
        ses ve görsel modellerindeki başarı oranı olarak yorumlanmaz.

        ## Metin girişini çalıştırma

        Dosyadaki metin okunur; öğrenci isteği mevcut beceriye bağlanır.
        Öneri mevcut soru kimliğini ve model tahminini içerir.
        '''),
        code('''
        live_examples = []
        text = (ROOT / "data/interactions/text_fractions.txt").read_text(encoding="utf-8")
        text_result = service.respond(text)
        live_examples.append({"modality": "text", **text_result})
        print("Öğrenci:", text)
        print("Öğretici:", text_result["reply"])
        print("Gerekçe:", text_result["recommendation"]["reason"])
        print("Geçmiş değişmedi:", text_result["history_unchanged"])
        '''),
        md('''
        ## Ses girişini çalıştırma

        Bu ses Microsoft Tolga ile üretilmiş Türkçe sentetik konuşmadır. WAV
        örnekleri gerçekten okunur ve 16000 Hz hızına dönüştürülür. Önceden
        eğitilmiş Whisper base sabit sürümde, Türkçe yazıya dökme göreviyle çalışır.
        Model bu projede yeniden eğitilmez. Ses oynatıcıdan dinlenebilir.
        '''),
        code('''
        display(Code(inspect.getsource(SpeechReader.read), language="python"))
        audio_path = ROOT / "data/interactions/audio_decimals.wav"
        display(Audio(filename=str(audio_path)))
        speech = SpeechReader(ROOT)
        recognized_audio = speech.read(audio_path)
        audio_result = service.respond(recognized_audio["text"])
        live_examples.append({"modality": "audio", "recognition": recognized_audio, **audio_result})
        print("Gerçek ASR çıktısı:", recognized_audio["text"])
        print("Öğretici:", audio_result["reply"])
        print("Örnekleme bilgileri:", {k: recognized_audio[k] for k in ["source_sample_rate", "model_sample_rate", "duration_seconds"]})
        '''),
        md('''
        Sesin beklenen cümlesi “Ondalık sayılarda toplama ve çıkarma çalışmak
        istiyorum.” ifadesidir. Gerçek ASR çıktısındaki sözcük hataları düzeltilmez.
        Beceri doğru seçilmiş olsa da bu, konuşmanın eksiksiz tanındığı anlamına
        gelmez. Kullanıcı tanınan metni görerek yeni bir girişle düzeltebilir.

        ## Görsel girişini çalıştırma

        Türkçe yazı içeren PNG, Windows OCR ile gerçekten okunur. Bu örnek
        basılı yazı tanımadır; el yazısı veya matematik şekli çözümleme içermez.
        '''),
        code('''
        image_path = ROOT / "data/interactions/image_integers.png"
        display(Image(filename=str(image_path), width=900))
        recognized_image = read_image(ROOT, image_path)
        image_result = service.respond(recognized_image["text"])
        live_examples.append({"modality": "image", "recognition": recognized_image, **image_result})
        print("Gerçek OCR çıktısı:", recognized_image["text"])
        print("OCR motoru ve dil:", recognized_image["engine"], recognized_image["language"])
        print("Öğretici:", image_result["reply"])
        (ROOT / "outputs/interactions/notebook_live_examples.json").write_text(
            json.dumps(live_examples, ensure_ascii=False, indent=2), encoding="utf-8")
        '''),
        md('''
        ## Dokuz kontrollü girdinin çıktıları

        Aynı üç istek üç giriş türünde hazırlanmıştır. Beklenen içerik yalnızca
        tanıma tamamlandıktan sonra ölçümde kullanılır. Ses veya OCR işlevlerine
        beklenen metin verilmez. Sözcük hata oranı harf, noktalama ve Türkçe
        karakter normalleştirmesinden sonra hesaplanır; ham çıktılar da saklanır.
        '''),
        code('''
        comparison = pd.DataFrame([{
            "Örnek": row["sample_id"], "Beklenen metin": row["evaluation"]["expected_text"],
            "Gerçek tanınan metin": row["recognized_text"],
            "Sözcük hata oranı %": 100*row["evaluation"]["normalized_word_error_rate"],
            "Beceri": row["intent"]["label"], "Beklenen beceri seçildi": row["evaluation"]["label_correct"],
            "Önerilen mevcut soru": row["recommendation"]["original_problem_id"] if row["recommendation"] else None,
        } for row in examples["results"]])
        with pd.option_context("display.max_colwidth", 100):
            display(comparison.round(2))
        display(pd.DataFrame(examples["modality_summary"]))
        '''),
        md('''
        Üç sentetik ses ve üç basılı görsel, genel ASR veya OCR değerlendirmesi
        değildir. Dokuz küçük girdinin doğru beceriye bağlanması ile sözcüklerin
        hatasız tanınması ayrı sonuçlardır. Yeni soru metni veya soru kimliği
        üretilmez; kaynakta özgün soru metni bulunmadığı için kimlik gösterilir.

        ## Belirsiz ve kapsam dışı isteklere yanıt
        '''),
        code('''
        for text in ["Matematik çalışmak istiyorum.", "Kesirlerde çarpma yapmak istiyorum.",
                     "Kesirler ve tam sayılarda toplama çalışalım."]:
            result = service.respond(text)
            print("Öğrenci:", text)
            print("Öğretici:", result["reply"])
            print("Öneri üretildi:", result["recommendation"] is not None)
            print()
        '''),
        md('''
        ## Kendi kısa girdini deneme

        `input_type` için `text`, `audio` veya `image` seçip bu hücreyi çalıştırın.
        Metinde `my_text`, dosyada `my_path` kullanılır. Ses 16 bit PCM WAV,
        0,2–30 saniye olmalı. OCR için Windows Türkçe tanıma dili kurulmalıdır.
        Tanınan metin de sonuçla birlikte gösterilir.
        '''),
        code('''
        input_type = "text"
        my_text = "Kesirlerde toplama yapmak istiyorum."
        my_path = ROOT / "data/interactions/audio_fractions.wav"

        if input_type == "text":
            recognized = {"text": my_text}
        elif input_type == "audio":
            recognized = speech.read(my_path)
        elif input_type == "image":
            recognized = read_image(ROOT, my_path)
        else:
            raise ValueError("Giriş türü text, audio veya image olmalı.")
        result = service.respond(recognized["text"])
        print("Alınan metin:", recognized["text"])
        print("Öğretici:", result["reply"])
        print("Cevap geçmişi değişmedi:", result["history_unchanged"])
        '''),
        md('''
        ## Güvenilirlik denetimi

        Öğrenci geçmişi, model ağırlıkları, davranış grupları ve referanslar
        sabit kalır. Metin, ses ve görsel verileri LKM içine verilmez; yalnızca
        çalışma isteğinin hangi mevcut beceriye ait olduğunu belirler. Öneri
        etkisini ölçen yeni bir öğrenme deneyi yapılmamıştır.

        Aşağıdaki denetim dosya özetlerini, hata ölçülerini ve önerilerin mevcut
        eğitim sorularına ait olduğunu yeniden kontrol eder. Ardından bu bölümün
        etkileşim testleri çalıştırılır; bütün projenin testleri sekizinci defterdedir.
        '''),
        code('''
        run_script("audit_interaction_examples.py")
        display(pd.Series(read_json("outputs/validation/interaction_integrity.json")))
        process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                                  "-p", "test_interactions.py", "-v"],
                                 cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        output = process.stdout + process.stderr
        (ROOT / "outputs/validation/interaction_test_output.txt").write_text(output, encoding="utf-8")
        print(output)
        process.check_returncode()
        '''),
    ], metadata={"kernelspec": {"display_name": "Knowledge Tracing (proje)", "name": "kt-project", "language": "python"},
                 "language_info": {"name": "python", "version": "3.13.2"}})
    nbf.validate(notebook)
    name = "07_Metin_Ses_ve_Gorsel_Etkilesimler.ipynb"
    nbf.write(notebook, ROOT / "notebooks" / name)
    print(name)


if __name__ == "__main__":
    main()
