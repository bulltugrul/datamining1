# Metin Ses ve Görsel Etkileşim Örnekleri

Üç giriş türü öğrencinin çalışmak istediği beceriyi ifade etmesini sağlar. Metin doğrudan, ses Whisper ile yazıya çevrilerek, görsel Windows OCR ile okunarak aynı beceri sınıflandırıcısına gider. Bilinen beceri tanınırsa mevcut KMPT modeli ve örnek öğrencinin değişmeyen geçmişiyle mevcut soru önerisi üretilir. Yeni soru kimliği veya doğru yanlış cevap kaydı oluşturulmaz.

## Küçük örneklerin kapsamı

Tam sayılarda toplama ve çıkarma, pozitif ondalık sayılarda toplama ve çıkarma, kesirlerde toplama ve çıkarma kapsanır. Bunlar ASSIST2009 içindeki sırasıyla 277, 278 ve 280 kaynak becerilerine bağlanır. Aynı üç farklı istek metin, ses ve görsel biçiminde sunularak toplam dokuz kontrollü girdi oluşturulmuştur. İfadeler sınıflandırıcı derlemine dahil edilmemiştir.

Sesler Microsoft Tolga ile üretilmiş sentetik Türkçe konuşmadır; gerçek öğrenci veya mikrofon kaydı değildir. Görseller Pillow ile basılmış kısa Türkçe yazı içerir. El yazısı, matematik şekli veya fotoğraftan denklem çözme işlevi yoktur. Girdi üretimi bir soru bankası hazırlama işi değildir.

## Metin sınıflandırma

Karakter temelli TF IDF üç ile beş karakterlik parçalar çıkarır; lojistik regresyon bilinen üç beceriyi öğrenir. Harf büyüklüğü, Türkçe karakterler ve noktalama normalleştirilir. Vektörleştirici ve sınıflandırıcı yalnızca 48 eğitim ifadesine uyarlandı. 20 doğrulama örneğiyle kabul eşiği 0.65 seçildi; eşit F1 halinde daha yüksek eşik alındı. En yakın eğitim ifadesiyle benzerlik alt sınırı 0.15 olarak önceden belirlendi.

Boş veya çok uzun istekler, birden çok beceri içeren açık ifadeler, çarpma ve bölme gibi desteklenmeyen işlemler için kapsam kuralları uygulanır. Düşük skorlu sonuçta da açıklama istenir. Kapsam dışı sınıf, bu kurallar ve çekimser kalma mekanizmasının çıktısıdır; dördüncü bir lojistik regresyon sınıfı olarak eğitilmemiştir. Bu sınırlı kontroller bütün dilsel belirsizlikleri çözen bir anlamlandırma sistemi değildir.

Ayrılmış 20 yazılmış test ifadesinde doğru etiket oranı yüzde 95.0, makro F1 0.9496. On iki kapsam içi örneğin biri düşük skor nedeniyle açıklama isteme sonucuna gitmiştir; sekiz kapsam dışı örnek reddedilmiştir. Bu değerler sınıflandırıcı ile kuralların birlikte işlev kontrolüdür. Küçük ve benzer sözcükler kullanan bu derlem, gerçek öğrencilerde genelleme ölçümü değildir. Sınıflandırma skoru öğrenci başarı yüzdesi veya kalibre edilmiş güven olarak sunulmaz.

## Ses ve görsel tanıma

ASR modeli openai/whisper-base, sabit sürümü e37978b90ca9030d5170a5c07aadb050351a65bb. Ağırlıklar hazırdır; bu projede ses modeli eğitilmez. WAV verisi gerçekten okunur, iki kanal varsa tek kanala indirilir, gerektiğinde 16000 Hz hızına yeniden örneklenir. Çok kısa, uzun veya sessiz kayıt ASR öncesinde reddedilir. Dil Türkçe, görev yazıya dökmedir. İlk model indirmesinden sonra işlem yereldir; API anahtarı gerekmez.

OCR, Windows.Media.Ocr.OcrEngine ve kurulu Türkçe tanıma diliyle çalışır. Sistem hazır modelini kullanır; ASSIST verisi üzerinde OCR eğitilmez. Kaynak görseldeki pikseller okunur, dosya adından veya beklenen metinden sonuç türetilmez. Windows OCR model dosyaları proje paketine dahil edilmez; sistemin kurulu sürümü kullanılır.

Beklenen metin yalnızca tanıma ve öneri bittikten sonra ölçüm için kullanılır. Tanınan metin elle düzeltilmez. Aşağıdaki hata ölçüsü Türkçe karakter, harf büyüklüğü ve noktalama normalleştirmesinden sonra sözcük ekleme, silme ve değiştirmelerini sayar. Ham metinler de ayrıca saklanır.

| Örnek | Gerçek tanıma çıktısı | Normalleştirilmiş sözcük hata oranı yüzde | Beklenen beceri seçildi |
|---|---|---|---|
| audio_integers | Negatif ve pozitif tam sayılarda toplama ve çıkarma alıştırmasın yapalım. | 10.00 | Evet |
| audio_decimals | 10 dalık sayılarda toplama ve çıkarmak çalışmak istiyorum. | 42.86 | Evet |
| audio_fractions | Paydaları farklı kesirlerde toplama ve çıkarma yaparken zorlanıyorum. | 0.00 | Evet |
| image_integers | Negatif ve pozitif tam sayılarda toplama ve çıkarma alıştırması yapalım. | 0.00 | Evet |
| image_decimals | Ondalık sayılarda toplama ve çıkarma çalışmak istiyorum. | 0.00 | Evet |
| image_fractions | Paydaları farklı kesirlerde toplama ve çıkarma yaparken zorlanıyorum. | 0.00 | Evet |

Üç ses dosyasındaki toplam sözcük hatasının toplam beklenen sözcük sayısına oranı yüzde 16.00. Dosya başına oranların basit ortalaması yüzde 17.62; bu iki hesap aynı ölçü değildir. Üç sentetik kayıt ve üç basılı görsel bir ASR veya OCR genel başarı testi sayılmaz. Dokuz girdinin tamamında beklenen beceri seçilmiş olması, konuşmanın hatasız tanındığı anlamına gelmez.

## Öğrenci takibine bağlantı

Örnek oturumun sonundaki 25 cevaplık geçmiş kullanılır. Tanınan beceri için sabit referans tahmini hesaplanır; ifade edilen beceri önceliğiyle mevcut soru kataloğundan öneri seçilir. Öneri gerekçesinde öğrencinin talebi açıkça belirtilir. Talep bir ölçme sorusuna verilmiş cevap değildir; geçmişe eklenmez, gözlenen doğru oranını ve davranış grubunu değiştirmez. Aynı geçmiş ve beceri için giriş türü tek başına farklı bilgi takibi tahmini üretmez. Metin, ses ve görsel özellikleri LKM içine verilmez; bunlar giriş dönüştürme ve beceri belirleme katmanıdır.

| İstenen beceri | Önerilen mevcut kaynak soru kimliği | Beceri referans tahmini yüzde | Soru doğru cevap tahmini yüzde |
|---|---|---|---|
| Tam sayılarda toplama ve çıkarma | 189214 | 43.92 | 65.84 |
| Pozitif ondalık sayılarda toplama ve çıkarma | 147305 | 32.61 | 65.49 |
| Kesirlerde toplama ve çıkarma | 58809 | 22.74 | 60.02 |

Kaynakta bu kimliklerin özgün soru metinleri yoktur; yeni soru metni uydurulmaz. Öneriler öğrenme kazanımı sağladığı ölçülmüş müdahaleler olarak sunulmaz.

## Çalıştırma ve çıktı kayıtları

`notebooks/07_Metin_Ses_ve_Gorsel_Etkilesimler.ipynb` girdileri, gerçekten tanınan metinleri, öğretici yanıtlarını ve ölçüleri birlikte gösterir. Son bölümde metin değiştirilebilir veya başka kısa PCM WAV ya da yazı içeren görsel dosyası denenebilir. Metin, ses ve görsel girdi işlevleri yalnızca dosya veya metin alır; beklenen cevap almaz.

Yeni kurulumda önce proje ortamı hazırlanır. Sonra `.venv/Scripts/python.exe scripts/download_interaction_model.py` açık ASR modelini sabit sürümde indirir. Hazır örnek girdiler `data/interactions/` içindedir; sesleri her çalıştırmada yeniden üretmek gerekmez. Sesleri yeniden üretmek için Windows Türkçe konuşma sesi, OCR için Windows Türkçe OCR dili gerekir. OCR bulunamazsa işlem açıklayıcı hatayla durur; tanıma çıktısı uydurulmaz. Ses tanıma ve metin sınıflandırma yerel Python ortamında çalışır; OCR yardımcısı Windows PowerShell kullanır.

`outputs/interactions/examples.json` dokuz girdinin dosya özetlerini, gerçek çıktıları, beklenen içerikle ayrı karşılaştırmasını ve önerileri saklar. `intent_metrics.json` derlem ayrımını ve eşik seçimini; `asr_model_provenance.json` model sürümünü ve dosya özetlerini kaydeder. Model lisansı Apache 2.0 olarak model kartında belirtilmiştir.

9 örneğin dosya özetleri, yönlendirmeleri, hata ölçüleri ve mevcut eğitim sorularıyla bağlantısı kontrol edildi. Bilgi takibi ağırlıkları, davranış grupları, referanslar ve öğrenci geçmişi aynı kaldı. Ana deney kaynakları korunuyor. Denetim `outputs/validation/interaction_integrity.json` içindedir.

## Yöntem kaynakları

- [Whisper base model kartı ve kullanım örneği](https://huggingface.co/openai/whisper-base)
- [Windows OCR motoru](https://learn.microsoft.com/en-us/uwp/api/windows.media.ocr.ocrengine)
- [Windows konuşma sentezleyicisi](https://learn.microsoft.com/en-us/uwp/api/windows.media.speechsynthesis.speechsynthesizer)
- [TF IDF vektörleştirici](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Lojistik regresyon](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
