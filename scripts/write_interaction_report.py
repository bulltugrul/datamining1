"""Gerçek giriş ve çıktı kayıtlarından etkileşim yöntemini ve ölçülerini yaz."""

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    results = read("outputs/interactions/examples.json")
    metrics = read("outputs/interactions/intent_metrics.json")
    config = read("configs/interactions.json")
    audit = read("outputs/validation/interaction_integrity.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    matrix = np.array(metrics["test_confusion_matrix"])
    labels = ["Tam sayılar", "Ondalık", "Kesir", "Kapsam dışı"]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max()/2 else "#152538", fontsize=14)
    ax.set_xticks(range(4), labels, rotation=15)
    ax.set_yticks(range(4), labels)
    ax.set(xlabel="Üretilen etiket", ylabel="Beklenen etiket",
           title="Yazılmış küçük ifade derleminde ayrılmış 20 örnek\nDenetimli sınıflandırıcı ve kapsam kuralları birlikte")
    for extension in ("png", "svg"):
        fig.savefig(ROOT / f"outputs/figures/interaction_intent_confusion.{extension}", dpi=170)
    plt.close(fig)
    lines = [textwrap.dedent(f"""\
        # Metin Ses ve Görsel Etkileşim Örnekleri

        Üç giriş türü öğrencinin çalışmak istediği beceriyi ifade etmesini sağlar. Metin doğrudan, ses Whisper ile yazıya çevrilerek, görsel Windows OCR ile okunarak aynı beceri sınıflandırıcısına gider. Bilinen beceri tanınırsa mevcut KMPT modeli ve örnek öğrencinin değişmeyen geçmişiyle mevcut soru önerisi üretilir. Yeni soru kimliği veya doğru yanlış cevap kaydı oluşturulmaz.

        ## Küçük örneklerin kapsamı

        Tam sayılarda toplama ve çıkarma, pozitif ondalık sayılarda toplama ve çıkarma, kesirlerde toplama ve çıkarma kapsanır. Bunlar ASSIST2009 içindeki sırasıyla 277, 278 ve 280 kaynak becerilerine bağlanır. Aynı üç farklı istek metin, ses ve görsel biçiminde sunularak toplam dokuz kontrollü girdi oluşturulmuştur. İfadeler sınıflandırıcı derlemine dahil edilmemiştir.

        Sesler Microsoft Tolga ile üretilmiş sentetik Türkçe konuşmadır; gerçek öğrenci veya mikrofon kaydı değildir. Görseller Pillow ile basılmış kısa Türkçe yazı içerir. El yazısı, matematik şekli veya fotoğraftan denklem çözme işlevi yoktur. Girdi üretimi bir soru bankası hazırlama işi değildir.

        ## Metin sınıflandırma

        Karakter temelli TF IDF üç ile beş karakterlik parçalar çıkarır; lojistik regresyon bilinen üç beceriyi öğrenir. Harf büyüklüğü, Türkçe karakterler ve noktalama normalleştirilir. Vektörleştirici ve sınıflandırıcı yalnızca {metrics['training_records']} eğitim ifadesine uyarlandı. {metrics['validation_records']} doğrulama örneğiyle kabul eşiği {metrics['threshold']:.2f} seçildi; eşit F1 halinde daha yüksek eşik alındı. En yakın eğitim ifadesiyle benzerlik alt sınırı {config['minimum_text_similarity']:.2f} olarak önceden belirlendi.

        Boş veya çok uzun istekler, birden çok beceri içeren açık ifadeler, çarpma ve bölme gibi desteklenmeyen işlemler için kapsam kuralları uygulanır. Düşük skorlu sonuçta da açıklama istenir. Kapsam dışı sınıf, bu kurallar ve çekimser kalma mekanizmasının çıktısıdır; dördüncü bir lojistik regresyon sınıfı olarak eğitilmemiştir. Bu sınırlı kontroller bütün dilsel belirsizlikleri çözen bir anlamlandırma sistemi değildir.

        Ayrılmış {metrics['test_records']} yazılmış test ifadesinde doğru etiket oranı yüzde {100*metrics['test_accuracy']:.1f}, makro F1 {metrics['test_macro_f1']:.4f}. On iki kapsam içi örneğin biri düşük skor nedeniyle açıklama isteme sonucuna gitmiştir; sekiz kapsam dışı örnek reddedilmiştir. Bu değerler sınıflandırıcı ile kuralların birlikte işlev kontrolüdür. Küçük ve benzer sözcükler kullanan bu derlem, gerçek öğrencilerde genelleme ölçümü değildir. Sınıflandırma skoru öğrenci başarı yüzdesi veya kalibre edilmiş güven olarak sunulmaz.

        ## Ses ve görsel tanıma

        ASR modeli {config['asr_model']}, sabit sürümü {config['asr_revision']}. Ağırlıklar hazırdır; bu projede ses modeli eğitilmez. WAV verisi gerçekten okunur, iki kanal varsa tek kanala indirilir, gerektiğinde 16000 Hz hızına yeniden örneklenir. Çok kısa, uzun veya sessiz kayıt ASR öncesinde reddedilir. Dil Türkçe, görev yazıya dökmedir. İlk model indirmesinden sonra işlem yereldir; API anahtarı gerekmez.

        OCR, Windows.Media.Ocr.OcrEngine ve kurulu Türkçe tanıma diliyle çalışır. Sistem hazır modelini kullanır; ASSIST verisi üzerinde OCR eğitilmez. Kaynak görseldeki pikseller okunur, dosya adından veya beklenen metinden sonuç türetilmez. Windows OCR model dosyaları proje paketine dahil edilmez; sistemin kurulu sürümü kullanılır.

        Beklenen metin yalnızca tanıma ve öneri bittikten sonra ölçüm için kullanılır. Tanınan metin elle düzeltilmez. Aşağıdaki hata ölçüsü Türkçe karakter, harf büyüklüğü ve noktalama normalleştirmesinden sonra sözcük ekleme, silme ve değiştirmelerini sayar. Ham metinler de ayrıca saklanır.

        | Örnek | Gerçek tanıma çıktısı | Normalleştirilmiş sözcük hata oranı yüzde | Beklenen beceri seçildi |
        |---|---|---|---|
        """).rstrip()]
    for row in results["results"]:
        if row["modality"] != "text":
            lines.append(f"| {row['sample_id']} | {row['recognized_text']} | {100*row['evaluation']['normalized_word_error_rate']:.2f} | {'Evet' if row['evaluation']['label_correct'] else 'Hayır'} |")
    audio = next(row for row in results["modality_summary"] if row["modality"] == "audio")
    lines.append(f"\nÜç ses dosyasındaki toplam sözcük hatasının toplam beklenen sözcük sayısına oranı yüzde {100*audio['pooled_normalized_word_error_rate']:.2f}. Dosya başına oranların basit ortalaması yüzde {100*audio['mean_normalized_word_error_rate']:.2f}; bu iki hesap aynı ölçü değildir. Üç sentetik kayıt ve üç basılı görsel bir ASR veya OCR genel başarı testi sayılmaz. Dokuz girdinin tamamında beklenen beceri seçilmiş olması, konuşmanın hatasız tanındığı anlamına gelmez.")
    lines.append("\n## Öğrenci takibine bağlantı\n\nÖrnek oturumun sonundaki 25 cevaplık geçmiş kullanılır. Tanınan beceri için sabit referans tahmini hesaplanır; ifade edilen beceri önceliğiyle mevcut soru kataloğundan öneri seçilir. Öneri gerekçesinde öğrencinin talebi açıkça belirtilir. Talep bir ölçme sorusuna verilmiş cevap değildir; geçmişe eklenmez, gözlenen doğru oranını ve davranış grubunu değiştirmez. Aynı geçmiş ve beceri için giriş türü tek başına farklı bilgi takibi tahmini üretmez. Metin, ses ve görsel özellikleri LKM içine verilmez; bunlar giriş dönüştürme ve beceri belirleme katmanıdır.\n\n| İstenen beceri | Önerilen mevcut kaynak soru kimliği | Beceri referans tahmini yüzde | Soru doğru cevap tahmini yüzde |\n|---|---|---|---|")
    for row in results["results"]:
        if row["modality"] == "text":
            rec = row["recommendation"]
            lines.append(f"| {rec['skill_name']} | {rec['original_problem_id']} | {row['reference_success_percent']:.2f} | {100*rec['predicted_probability']:.2f} |")
    lines.append(textwrap.dedent(f"""\

        Kaynakta bu kimliklerin özgün soru metinleri yoktur; yeni soru metni uydurulmaz. Öneriler öğrenme kazanımı sağladığı ölçülmüş müdahaleler olarak sunulmaz.

        ## Çalıştırma ve çıktı kayıtları

        `notebooks/07_Metin_Ses_ve_Gorsel_Etkilesimler.ipynb` girdileri, gerçekten tanınan metinleri, öğretici yanıtlarını ve ölçüleri birlikte gösterir. Son bölümde metin değiştirilebilir veya başka kısa PCM WAV ya da yazı içeren görsel dosyası denenebilir. Metin, ses ve görsel girdi işlevleri yalnızca dosya veya metin alır; beklenen cevap almaz.

        Yeni kurulumda önce proje ortamı hazırlanır. Sonra `.venv/Scripts/python.exe scripts/download_interaction_model.py` açık ASR modelini sabit sürümde indirir. Hazır örnek girdiler `data/interactions/` içindedir; sesleri her çalıştırmada yeniden üretmek gerekmez. Sesleri yeniden üretmek için Windows Türkçe konuşma sesi, OCR için Windows Türkçe OCR dili gerekir. OCR bulunamazsa işlem açıklayıcı hatayla durur; tanıma çıktısı uydurulmaz. Ses tanıma ve metin sınıflandırma yerel Python ortamında çalışır; OCR yardımcısı Windows PowerShell kullanır.

        `outputs/interactions/examples.json` dokuz girdinin dosya özetlerini, gerçek çıktıları, beklenen içerikle ayrı karşılaştırmasını ve önerileri saklar. `intent_metrics.json` derlem ayrımını ve eşik seçimini; `asr_model_provenance.json` model sürümünü ve dosya özetlerini kaydeder. Model lisansı Apache 2.0 olarak model kartında belirtilmiştir.

        {audit['examples_checked']} örneğin dosya özetleri, yönlendirmeleri, hata ölçüleri ve mevcut eğitim sorularıyla bağlantısı kontrol edildi. Bilgi takibi ağırlıkları, davranış grupları, referanslar ve öğrenci geçmişi aynı kaldı. Ana deney kaynakları korunuyor. Denetim `outputs/validation/interaction_integrity.json` içindedir.

        ## Yöntem kaynakları

        - [Whisper base model kartı ve kullanım örneği](https://huggingface.co/openai/whisper-base)
        - [Windows OCR motoru](https://learn.microsoft.com/en-us/uwp/api/windows.media.ocr.ocrengine)
        - [Windows konuşma sentezleyicisi](https://learn.microsoft.com/en-us/uwp/api/windows.media.speechsynthesis.speechsynthesizer)
        - [TF IDF vektörleştirici](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
        - [Lojistik regresyon](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
        """).rstrip())
    (ROOT / "docs/metin_ses_gorsel_ornekleri.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("docs/metin_ses_gorsel_ornekleri.md")


if __name__ == "__main__":
    main()
