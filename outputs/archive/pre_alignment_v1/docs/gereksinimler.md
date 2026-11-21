# Akıllı Öğretim Projesi Gereksinimleri

Proje, matematik öğrenme etkileşimlerinde prompt tuning ile bilgi takibini inceler. Çalışan Python uygulaması, öğrenme görevlerini açıklayan blok diyagramı ve kod ile çıktıları açıklayan rapor tek arşivde teslim edilir. Kullanıcı ile kararlaştırılan kapsam, ASSIST2009 üzerinde yeniden kurulmuş deneyler ve mevcut kayıtları adım adım işleyen örnek oturumdur.

## Hocanın istediği işlevler

| Kimlik | Gereksinim | Karşılanacağı bölüm | Kabul ölçütü |
|---|---|---|---|
| H01 | Makaleye dayalı akıllı öğretim prototipi | Bilgi takibi modeli | LKM ön eğitimi ve prompt tabanlı cevap tahmini çalışan kodla gösterilir. |
| H02 | Öğrenci başarısının yüzdeyle izlenmesi | Beceri göstergeleri | Sabit referans soruların tahmini doğru cevap olasılıkları ortalanır; gözlenen doğru cevap oranı ayrı gösterilir. |
| H03 | Yüzdenin zaman içindeki değişimi | Örnek oturum | Her cevap sonrasında göstergeler güncellenir; sıra, gösterim zamanı ve kaynak olay zamanı birbirinden ayrılır. |
| H04 | Öğrenme örüntülerinin bulunması | Davranış gruplama | Benzer yakın dönem geçmişleri gruplandırılır; grupların özellikleri açıklanır. |
| H05 | Denetimli ve denetimsiz görevlerden oluşan diyagram | Öğrenme blokları diyagramı | Her öğrenme bloğunun görevi, öğrenme türü, girdisi ve çıktısı tanımlanır. |
| H06 | Ayrıntılı öğrenci ve öğretici etkileşim senaryosu | Örnek oturum ve küçük etkileşim örnekleri | Tahmin, cevap, güncelleme ve öneri sırası açıklanır. Metin, ses ve görsel için çalışan küçük örnekler sunulur. |
| H07 | Python uygulaması ve açıklamalı rapor | Kod defterleri ve rapor | Kod ve çıktı bölümleri açıklanır; ilk prototip duyurusundaki beklentiye uygun ekran görüntüleri hazırlanır. |
| H08 | Grup halinde teslim | Son teslim paketi | Kod ve rapor tek ZIP veya RAR dosyasında; tüm üyelerin adı, soyadı ve numarası dahil edilir. |

Son teslim tarihi 23 Kasım 2026. Grup adına tek kişi, hocanın belirttiği bağlantı üzerinden gönderir. Ayrı web veya mobil uygulama zorunluluğu duyurularda belirtilmemiştir.

## Birlikte kararlaştırılan kapsam

- Ham ASSIST2009 dosyası kullanılabilir. Önceki defterler, eğitilmiş ağırlıklar ve raporlanmış sonuçlar bu uygulamaya alınmaz.
- Tek ders matematik; veri kümesindeki beceriler ve farklı soru zorlukları kullanılır.
- Ortak ön eğitimden sonra promptsuz uyarlama, KPT ve KMPT karşılaştırılır. Her koşul üç bağımsız başlangıçla değerlendirilir.
- Eğitim, doğrulama ve test ayrımı öğrenci bazında yapılır. Test verisi model, eşik, referans soru kümesi veya küme sayısı seçmek için kullanılmaz.
- Öğrenci göstergeleri cevap geldikçe güncellenir. Bir göstergenin artması zorunlu değildir.
- Gruplar yakın dönem cevap davranışlarını tanımlar; kalıcı öğrenme tarzı veya öğrenci niteliği olarak yorumlanmaz.
- Soru ve çalışma önerileri, model skorlarını kullanan açıklanabilir kurallarla üretilir. Bu seçim kuralları ayrı bir öğrenilmiş önerici model olarak sunulmaz.
- Metin, ses ve görsel için gereken kadar küçük örnek içerik hazırlanır. Bu örnekler kapsamlı yeni soru bankasına dönüşmez.
- Kapsamlı soru bankası hazırlama, yeni yazılmış sorular için model uyarlama ve eski projeden kod taşıma kapsam dışındadır.

## Ölçüm tanımları

Beceri bazında tahmini başarı yüzdesi, ilgili beceriye ait sabit referans soru kümesi üzerinde güncel öğrenci geçmişiyle hesaplanan doğru cevap olasılıklarının ortalamasının yüz katıdır. Bu kümenin üyeleri ve ağırlıkları oturum boyunca sabit kalır. Çok becerili bir soru her ilgili becerinin referans kümesine girebilir; bu üyelik raporlanır.

Gözlenen doğru cevap oranı, öğrencinin tanımlı aralıktaki gerçek cevap sonuçlarından hesaplanır. Modelin tahmin doğruluğu ve AUC ise bağımsız test örnekleri üzerindeki model değerlendirmesidir. Üç ölçü ayrı isimlerle sunulur.

Örnek oturum eski etkileşimleri yeniden oynatır. Kaynak dosyada takvim zaman damgası bulunmadığından oturumun gösterim saati kaydedilir; grafiğin birincil ekseni etkileşim sırasıdır. Gösterim saniyeleri öğrencinin geçmişteki öğrenme veya unutma süresi olarak yorumlanmaz. Bu örnek, kayıt işleme ve gösterge güncelleme davranışını gösterir.

## Veri ve deney güvenilirliği

| Kontrol | Beklenen durum |
|---|---|
| Öğrenci ayrımı | Eğitim, doğrulama ve test öğrenci kümelerinin kesişimleri boş. |
| Gelecek cevapların gizlenmesi | Tahmin yalnızca geçmişi, hedef soruyu ve bilinen soru özelliklerini kullanır. |
| Çok becerili kayıtlar | Aynı gerçek cevabın kopyaları geçmişe ek cevap gibi sızmaz. |
| Soru zorluğu | Eğitim kayıtlarındaki hata oranından hesaplanır; doğrulama ve test cevapları kullanılmaz. |
| Bilinmeyen kimlikler | Ayrılmış veride eğitimde görülmeyen soru veya beceriler için davranış ve kapsama oranı raporlanır; kayıtlar sessizce elenmez. |
| Promptların etkisi | Eğitilen promptlara gradyan ulaştığı ve tahmin konumunun prompt temsillerini görebildiği doğrulanır. |
| Referans soru kümesi | Test cevaplarına bakılmadan seçilir; oturum boyunca değişmez. |
| Tekrarlanabilirlik | Ham dosya özeti, öğrenci ayrımı, başlangıçlar, ayarlar ve paket sürümleri kaydedilir. |
| Küçük etkileşim örnekleri | Girdi dosyası, beklenen içerik ve gerçekten üretilen çıktı birlikte gösterilir. |

## Kaynaklar

Birincil proje kaynağı Zhu, Qiu ve Zhou'nun 2026 tarihli [Prompt tuning for improving knowledge tracing in intelligent tutoring](https://link.springer.com/article/10.1007/s10618-026-01208-9) makalesidir. Tam metin kullanıcı tarafından sağlanmıştır. LKM, RTM, FBO, PromEnc ve KPT ile KMPT tasarımı bu kaynağa dayanır; kapasite ve uygulama farklılıkları raporda açıklanır.

Görev ve teslim koşulları kullanıcının paylaştığı 23 Ekim, 30 Ekim, vize ve vize ipucu duyurularından alınmıştır. Bağımsız duygu analizi ve paralel optimizasyon ödevleri bu gereksinimlere dahil edilmez.

Veri kaynağı [ASSISTments 2009 ile 2010 veri açıklaması](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data) ve [düzeltilmiş Skill Builder dosyası](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) sayfalarıdır.
