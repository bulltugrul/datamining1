"""Kaynak PDF, eşitlikler ve eski/yeni kaynaklarla denetim kararlarını kaydet."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MATRIX = [
    ("Dikkat eğitimde", "3.2,3.4; s.8,13–14", "LKM nedensel, PromEnc iki yönlü", "İki modül de iki yönlü", "Düzeltildi"),
    ("Dikkat tahminde", "3.5; s.14", "LKM nedensel, PromEnc iki yönlü", "İki modül de nedensel", "Düzeltildi"),
    ("Şablon sırası", "Denklem10–13; Şekil6–7", "Pseudo tokenlar başta; maske sonda", "e,c,MASK_r,T ve MASK_e,c,r,T", "Düzeltildi; başta prompt düzeni ayrı kontrol"),
    ("Çıktı konumu", "3.4–3.5; s.14", "Daima son konum", "Cevapta şablonun 2., soruda 0. indisi", "Düzeltildi; indisler sıfır tabanlı"),
    ("Şablonun soru temsili", "Denklem12–13", "Soru temsiline hedef zorluğu ekleniyor", "Şablonda e,c,r; geçmişte zorluk korunuyor", "Düzeltildi"),
    ("FBO negatifleri", "3.2; s.9", "Başka öğrencinin ikinci yarısı", "Bütün yarım parça havuzundan özgün doğru çift hariç rastgele çift", "Düzeltildi"),
    ("RTM bozulması", "3.2; s.9", "Seçilen konumda soru ve cevap bağımsız bozuluyor", "Konum başına ortak 80/10/10 işlemi; eşlenmiş soru ve cevap", "Konum temelli tarifin açık uygulaması"),
    ("Kayıp ölçekleme", "Denklem3–7", "RTM token ortalaması ve FBO dizi ortalaması toplanıyor", "Bütün kayıp toplamı aynı dizi sayısına bölünüyor", "Denklem toplam ölçeği uygulandı; yazarın kod ölçeği doğrulanmadı"),
    ("BERT katman düzeni", "3.2; s.8–9", "Ön normalizasyon ve son ek normalizasyon", "BERT tarzı son normalizasyon; son ek norm yok", "Düzeltildi"),
    ("En uzun geçmiş, dropout, optimizer", "4.2; s.18", "128,0.2,AdamW", "200,0.4,Adam", "Düzeltildi"),
    ("LKM parametre güncellemesi", "Algoritma1 adım24; s.15", "LKM ve prompt eğitiliyor", "LKM ve prompt eğitiliyor", "Korundu; makale LKM dondurma şartı koymuyor"),
    ("KMPT örnek oranı", "3.4; s.14", "80% cevap,20% soru", "80% cevap,20% soru", "Korundu; gerçek sayımlar kaydediliyor"),
    ("Çok kavramlı olay", "Denklem2 ile 4.2", "Tek gerçek cevap, kavram ortalaması", "Tek gerçek cevap, kavram ortalaması", "Kasıtlı veri işleme farkı; kopya cevap sızıntısı önleniyor"),
    ("Zorluk kaynağı", "4.2; s.18", "Yalnızca eğitim hata oranı", "Yalnızca eğitim hata oranı", "Kasıtlı koruma; tutulmuş cevaplar özellik hesabına alınmıyor"),
    ("Model kapasitesi", "4.2; s.18", "LKM2/128/4,PromEnc1", "LKM2/128/4,PromEnc1", "Ölçekli uygulama; makale8/256/8,PromEnc4"),
    ("Değerlendirme", "4.2; s.18", "Öğrenci bazlı70/15/15,üç başlangıç", "Aynı öğrenci ayrımı ve üç başlangıç", "Kasıtlı proje düzeni; makale80/20 ve5kat"),
    ("Promptsuz karşılaştırma", "3.4;4.1", "Ortak LKM ve yeni MLP başlığı", "Tam hedef etkileşimi ve yeni MLP kullanan proje içi kontrol", "Hedef zorluğunu da görür; makaledeki12model benchmarkının birebir karşılığı değil"),
    ("Belirtilmeyen alt ayrıntılar", "3.2–3.4", "Yerel Transformer,konum temsilleri ve doğrusal RTM başlıkları", "Açık kaynak yerel uygulama tercihleri", "Yazar kodu doğrulanmadığı için tam yazılım eşitliği iddia edilmez"),
    ("Uyarlama örneklerinin kaynağı", "Algoritma1 adım22; s.15", "Gerçek öğrenci geçmişlerinin her hedef ön eki", "Gerçek öğrenci ön ekleri korunuyor", "Algoritma artırılmış X' kümesine işaret eder; hedef ve geçmiş kurma ayrıntısı açıklanmıyor. Tam Algoritma1 tekrarı değil"),
]


def main():
    folder = ROOT / "outputs/method_audit"
    folder.mkdir(parents=True, exist_ok=True)
    local_pdf = ROOT / "references/paper_plkt.pdf"
    pdf = local_pdf if local_pdf.exists() else Path("C:/Users/unalm/Downloads/s10618-026-01208-9 (4).pdf")
    matrix = [{"component": row[0], "paper_location": row[1], "legacy": row[2],
               "corrected": row[3], "decision": row[4]} for row in MATRIX]
    record = {"paper_path": str(pdf), "paper_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
              "paper_doi": "10.1007/s10618-026-01208-9", "reviewed_pdf_pages": [8,9,10,11,12,13,14,15,18,22],
              "matrix": matrix,
              "critical_ambiguity": "With Eq12–13 tail pseudo tokens and strict causal masks in BOTH encoders at prediction, response MASK cannot depend on tail pseudo tokens. Tested as a literal reading, not silently repaired.",
              "freeze_interpretation": "Algorithm1 step24 tunes the pre-trained LKM; the article does not specify a PEFT-only frozen-LKM requirement.",
              "test_limitation": "The fixed test holdout was viewed in v1. V2 is a methodological re-evaluation, not a new untouched confirmatory experiment."}
    (folder / "audit_matrix.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Makale ile Yöntem Uyumu Denetimi", "",
        "Denetim makalenin yöntem eşitlikleri, şekilleri ve Algoritma 1 ile çalışan kodu karşılaştırır. Yeni uygulama çekirdek yöntem farklarını giderir; ölçekli model kapasitesi ve öğrenci bazlı değerlendirme düzeni korunur. Önceki kaynaklar ve sonuçlar ayrı sürüm olarak saklanır.", "",
        "| Bileşen | Makaledeki yer | Önceki uygulama | Düzeltilen uygulama | Karar |",
        "|---|---|---|---|---|"]
    lines += ["| "+" | ".join(row)+" |" for row in MATRIX]
    lines += ["", "## Şablon sırası ile nedensel tahmin arasındaki gerilim", "",
        "Denklem 12 ve 13 pseudo tokenları maskeden sonra yerleştirir. Bölüm 3.5 ise tahmin sırasında hem LKM hem PromEnc dikkatini nedensel hale getirir. Bu iki tarif birlikte katı biçimde uygulandığında maskeli cevap konumuna sonraki pseudo tokenlardan bilgi akamaz. Bu sonuç makalenin yazarlarının koduna ilişkin bir iddia değildir; sağlanan metnin doğrudan uygulamasından çıkan ve kodda sınanan bir özelliktir. Yazar kodu veya ek açıklama olmadan farklı bir dikkat maskesi varsaymak birebir uyum diye sunulamaz.", "",
        "Bu nedenle ana KPT ve KMPT deneyleri metindeki sıra ve dikkatle çalışır. Ek KMPT prefix kontrolü yalnızca şablon sırasını değiştirip pseudo tokenları maskeden önce koyar. Bu kontrol makalenin yönteminin yerine geçirilmez. Eğitimde pseudo token gradyanı ve tahminde pseudo token değişikliğine duyarlılık ayrı ölçülür. Eğitimde öğrenilmiş ortak ağırlıklar nedeniyle prompt eğitimi dolaylı etki bırakabilir; doğrudan tahmin duyarlılığının sıfır olması bütün eğitimin etkisiz olduğu anlamına gelmez.", "",
        "Öğrenme adımı temelli blok maske ayrıca tanısal kontrol olarak uygulanır: geçmiş nedensel kalır, aynı hedefin bilinen/MASK şablon tokenları ortak erişir. Altı ek tahmin değerlendirmesi aynı sabit ağırlıkları kullanır; yeni eğitim veya bu maske için dönem seçimi yapılmaz. Çalışan prototipin blok politikası testten önce configs/inference_interpretation.json ile kaydedilmiştir; KMPT başlangıcı yalnızca blok doğrulama BCE değeriyle seçilir. Katı token yorumu on iki ana eğitimde ayrı tutulur. Açık yazar maskesi doğrulanmadığından iki yorumdan hiçbiri yazarların kesin yazılım tercihi diye sunulmaz.", "",
        "## Uyarlama verisi ve Algoritma 1 sınırı", "",
        "Algoritma 1 adım22 uyarlama döngüsünde artırılmış X' kümesini kullanır. Bölüm 3.4 ise her öğrenme dizisi için şablon kurmayı tarif eder; FBO negatiflerinin, SEP konumlarının ve RTM bozulmalarının downstream hedef ve geçmişlerine nasıl dönüştürüleceği açık değildir. Bu uygulama ön eğitimde artırılmış çiftleri, downstream aşamada gerçek öğrencinin her cevabından önceki gözlenmiş öneki kullanır. Bu, bir varsayımı sessizce tamamlamak yerine korunan ve açıkça kaydedilen protokol farkıdır. Eşitlik ve dikkat düzeltmeleri, Algoritma 1'in bütün veri akışının birebir yeniden üretildiği iddiası değildir.", "",
        "## LKM dondurma konusunda karar", "",
        "Genel prompt tuning terimi tek başına LKM dondurmayı gerektiren bir uygulama şartı sayılmaz. Bu makalede Algoritma 1 adım 24 ön eğitimli LKM'nin uyarlanmasını söyler; sabitlenmesini açıkça şart koşmaz. LKM'nin eğitilmesi korunmuştur. Sonuçlar yalnızca prompt parametrelerini eğiten saf PEFT karşılaştırması olarak adlandırılmaz.", "",
        "## Korunan deney farkları", "",
        "LKM iki katman, 128 boyut ve dört başlık; PromEnc bir katmandır. Makale sırasıyla sekiz katman, 256 boyut, sekiz başlık ve dört PromEnc katmanı kullanır. Downstream toplu işlem sayısı yerel eğitim süresi ve bellek ölçümüyle 512 olarak belirlenmiştir; makalede 64'tür. Bunlar yeni uygulamayı ölçekli bir yöntem uygulaması yapar; birebir deney tekrarı iddiası kurulmaz.", "",
        "Üç başlangıç 17,42,2026; sabit öğrenci ayrımı ve aynı hedefler kullanılır. Yeni ön eğitim bütün yöntemlerde aynı başlangıç içinde paylaşılır. Model ve dönem seçimi doğrulama BCE değeriyledir. Tüm eğitimler bitip kontrol noktaları ve ayarlar kilitlenmeden yeni test sonuçları hesaplanmaz. Önceki sürümde aynı test kümesi zaten görülmüştür; yeni sayılar tamamen dokunulmamış yeni bir doğrulayıcı test gibi sunulmaz. Eski ve yeni sürüm arasında birden çok değişiklik olduğundan başarı farkı tek bir düzeltmeye atfedilmez. Prefix kontrolü ise yeni KMPT ile yalnızca şablon yerleşimi farkını inceler.", "",
        "Ön eğitim kaybı göreli iyileşmesi ve sonlu dönem bütçesi kaydedilir. Bütçenin bitmesi yakınsamanın kanıtı olarak sunulmaz. Tek ders/veri kümesi kapsamı korunmuştur; makaledeki dört veri kümesi ve tüm karşılaştırma modelleri bu projede yeniden üretilmez.", "",
        "## Kaynak", "",
        "Zhu, Qiu ve Zhou, [Prompt tuning for improving knowledge tracing in intelligent tutoring](https://doi.org/10.1007/s10618-026-01208-9), kullanıcı tarafından sağlanan 30 sayfalık tam metin. Özellikle s.9–15 ve18. PDF içerik özeti ile madde bazlı kayıt outputs/method_audit/audit_matrix.json içindedir."]
    (ROOT / "docs/makale_yontem_uyumu.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"reviewed_components": len(matrix), "paper_sha256": record["paper_sha256"]}))


if __name__ == "__main__":
    main()
