# Küçük Eğitim Denemesi Sonuçları

Bu belge önceki uygulama sürümünü açıklar. Yöntem denetimi sonrasındaki güncel uygulama için [makale uyumu](makale_yontem_uyumu.md) ve [düzeltilmiş sonuçlar](duzeltilmis_yontem_sonuclari.md) esas alınmalıdır. Önceki sayılar tarihsel kayıt olarak korunmuştur.

LKM ön eğitimi ve üç uyarlama yöntemi CUDA üzerinde çalıştı. Kaydedilen ağırlıklar yeniden yüklendiğinde doğrulama ölçüleri yeniden üretildi. Bu deneme tek başlangıç ve sınırlı örnek kullanır; ana bilimsel karşılaştırma sonucu değildir.

## Deneme kapsamı

Başlangıç 17, eğitim öğrencisi üst sınırı 96, doğrulama öğrencisi üst sınırı 32 olarak seçildi. Uyarlamada 2048 eğitim hedefi ve 512 doğrulama hedefi kullanıldı. LKM üç, uyarlamalar en fazla dört dönem çalıştı. Bağımsız test değerlendirmesi yapılmadı.

Model iki katman ve 128 boyutlu temsiller kullanıyor. Ekran kartı RTX 4060; karma hassasiyet bfloat16. Aygıt üzerinde toplu geçmiş oluşturma kullanan denemede en yüksek ayrılan PyTorch GPU belleği yaklaşık 437,5 MiB oldu.

## Doğrulama ölçüleri

En iyi dönem en düşük doğrulama BCE değeriyle seçildi. Diğer ölçüler aynı döneme aittir.

| Yöntem | Seçilen dönem | Doğrulama BCE | Doğrulama AUC | Doğrulama doğruluğu |
|---|---|---|---|---|
| Promptsuz uyarlama | 2 | 0.6125 | 0.7425 | 0.6934 |
| KPT | 2 | 0.6199 | 0.7247 | 0.6582 |
| KMPT | 1 | 0.6527 | 0.6542 | 0.6191 |

Bu küçük örnekte prompt yöntemleri promptsuz yöntemi geçmiyor. Dönemler ilerledikçe doğrulama kaybının artması, sınırlı örnekte aşırı uyuma işaret ediyor. Bu sıralama nihai sonuç olarak genellenmez. Ana deney aynı ayrımı koruyarak bütün eğitim ve doğrulama öğrencileriyle ve üç başlangıçla tamamlandı; [ana deney sonuçları](ana_deney_sonuclari.md) ayrı sunuluyor.

## Güvenilirlik kontrolleri

- Üç yöntemin ortak LKM başlangıcı içerik özetiyle doğrulandı.
- Hedef etiket değiştirildiğinde tahmin değişmedi.
- Gelecekteki bir etkileşim değiştirildiğinde daha önceki nedensel temsil değişmedi.
- Farklı uzunluktaki geçmişlerle aynı toplu girdiye alınan öğrencinin tahmini dolgu nedeniyle değişmedi.
- KPT ve KMPT soft promptları gradyan aldı ve güncellendi.
- Yardımcı soru tahmini değerlendirme sırasında kapalı kaldı.
- RTM ve FBO ön eğitimi eğitim öğrencileriyle sınırlı kaldı.
- Kayıtlı ağırlıklar yeniden yüklendiğinde doğrulama ölçüleri tekrar elde edildi.
- Veri, model ve çalışma defteri başlatıcısı dahil 13 otomatik kontrol geçti.

## Ana deney için hız ve bellek

Yalnızca eğitim girdileriyle yapılan KPT ölçümünde 128, 512 ve 1024 toplu örnek sayıları sınandı. Ölçülen hızlar sırasıyla yaklaşık 5471, 6778 ve 6953 hedef/saniye; en yüksek ayrılan bellekler 381, 1265 ve 2452 MiB oldu. Bunlar kısa donanım ölçümleridir; bütün deneyin bitiş süresini göstermez.

Ana deneyde 512 seçildi: 1024'e göre hız farkı küçükken bellek kullanımı yaklaşık yarıdır. Bu seçim test ölçülerine dayanmaz. Bütün karşılaştırma yöntemleri aynı toplu örnek sayısını kullanacak.

## Kayıtlar

Pilot ayarları, ağırlıklar ve ham ölçüler `outputs/pilot/lkm_128_device_batches` altında bulunur. Grafikler `outputs/figures/pilot_pretraining.png` ve `outputs/figures/pilot_validation_bce.png` dosyalarıdır. Yeniden yükleme denetimi `outputs/validation/pilot_integrity.json`, test çıktısı `outputs/validation/model_test_output.txt` altında saklanır.
