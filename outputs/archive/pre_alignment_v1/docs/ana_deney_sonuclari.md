# Üç Başlangıçlı Ana Deney Sonuçları

Promptsuz uyarlama, KPT ve KMPT; aynı öğrenci ayrımında, her başlangıç için ortak LKM ağırlıklarından başlayarak karşılaştırıldı. En iyi dönem yalnızca doğrulama BCE değerinden seçildi. Seçilen ağırlıklar bağımsız testte değerlendirildi.

## Veri ve deney düzeni

Her koşul 194663 eğitim ve 38792 doğrulama hedefi kullandı. Testte 548 öğrenciye ait 44791 cevap hedefi bulunuyor. Her öğrencinin ilk etkileşimi başlangıç geçmişi olarak kullanıldığı için hedef sayıları ham etkileşim sayılarından daha düşüktür.

Başlangıçlar 17, 42 ve 2026. Aynı başlangıç içindeki üç yöntemin LKM içerik özetleri eşleşiyor. Farklı başlangıçların LKM özetleri farklı. Veri hazırlama ve model kaynaklarının eğitim anındaki içerik özetleri doğrulandı.

## Birincil test sonuçları

Değerler üç başlangıcın ortalaması ve örnek standart sapmasıdır. AUC ile doğrulukta yüksek, BCE değerinde düşük sonuç daha iyidir. Ölçüler etkileşim ağırlıklıdır; öğrenci başına eşit ağırlıklı ortalamalar değildir.

| Yöntem | Test AUC | Test doğruluğu yüzde | Test BCE |
|---|---|---|---|
| Promptsuz uyarlama | 0.7672 ± 0.0029 | 73.27 ± 0.26 | 0.5461 ± 0.0036 |
| KPT | 0.7586 ± 0.0023 | 72.57 ± 0.39 | 0.5506 ± 0.0030 |
| KMPT | 0.7558 ± 0.0025 | 72.37 ± 0.22 | 0.5632 ± 0.0040 |

Bu standart sapma, sabit bir öğrenci ayrımında başlangıçlar arasındaki değişkenliği gösterir. Öğrenci evrenine ilişkin güven aralığı veya istatistiksel anlamlılık sonucu değildir.

## Her başlangıcın sonucu

| Başlangıç | Yöntem | Seçilen dönem | Doğrulama BCE | Test AUC | Test BCE | Test doğruluğu yüzde |
|---|---|---|---|---|---|---|
| 17 | Promptsuz uyarlama | 1 | 0.5501 | 0.7690 | 0.5434 | 73.53 |
| 17 | KPT | 1 | 0.5636 | 0.7612 | 0.5538 | 72.99 |
| 17 | KMPT | 1 | 0.5715 | 0.7562 | 0.5625 | 72.51 |
| 42 | Promptsuz uyarlama | 1 | 0.5562 | 0.7639 | 0.5502 | 73.00 |
| 42 | KPT | 1 | 0.5540 | 0.7572 | 0.5478 | 72.49 |
| 42 | KMPT | 2 | 0.5745 | 0.7531 | 0.5675 | 72.11 |
| 2026 | Promptsuz uyarlama | 1 | 0.5522 | 0.7689 | 0.5447 | 73.26 |
| 2026 | KPT | 1 | 0.5581 | 0.7575 | 0.5501 | 72.22 |
| 2026 | KMPT | 2 | 0.5660 | 0.7580 | 0.5597 | 72.48 |

## Sonuçların yorumu

Bu uygulama ve sabit veri ayrımında en yüksek ortalama test AUC değeri Promptsuz uyarlama yönteminde elde edildi.

KPT ile promptsuz uyarlama arasındaki aynı başlangıçlar üzerinden hesaplanmış AUC farkı -0.0086; bu farkın örnek standart sapması 0.0025.

KMPT ile promptsuz uyarlama arasındaki aynı başlangıçlar üzerinden hesaplanmış AUC farkı -0.0114; bu farkın örnek standart sapması 0.0011.

Karşılaştırma yöntemlerin toplam etkisini ölçer. Şablonlar, PromEnc, başlık kullanımı ve KMPT'nin yardımcı görevi birlikte değişir. Sonuç yalnızca soft promptlara atfedilemez. Üç başlangıç ve tek öğrenci ayrımı yöntemin tüm veri kümelerinde üstünlüğünü kanıtlamaz.

Model kapasitesi, çok becerili kayıtların korunması, dikkat düzeni ve veri ayrımı makaledeki deneyin birebir kopyası değildir. Bu farklılıkların tekil etkileri burada ayrı ablasyonlarla ölçülmedi.

## Eğitimde görülmeyen sorular

Test hedeflerinin 304 tanesinde eğitimde görülmeyen soru kimliği var. Bu hedefler birincil değerlendirmeden çıkarılmadı. Bilinen soru alt grubunun AUC sonuçları aşağıdadır.

| Yöntem | Bilinen sorularda AUC ortalama ve standart sapma |
|---|---|
| Promptsuz uyarlama | 0.7676 ± 0.0029 |
| KPT | 0.7590 ± 0.0023 |
| KMPT | 0.7563 ± 0.0025 |

## Örnek oturumda kullanılacak model

Örnek oturum için önceden belirlenen yöntem KMPT'dir. Üç KMPT ağırlığı içinden yalnızca doğrulama BCE değeriyle başlangıç 2026 seçildi. Bu seçimde test skorları kullanılmadı. Bu tercih, KMPT'nin testte en iyi yöntem olduğu iddiası değildir.

## Çalışma maliyeti ve kayıtlar

Kayıtlı toplam deney süresi 27.31 dakika; en yüksek ayrılan PyTorch GPU belleği 1315.3 MiB. Aygıt NVIDIA GeForce RTX 4060, hassasiyet bfloat16. Süreye uyarlama dönemlerindeki doğrulama işlemleri dahildir.

Ayarlar, ağırlıklar, dönem geçmişleri ve örnek bazındaki test tahminleri outputs/main/three_seed_v1 altında saklanır. Yeniden hesaplanan ölçüler ve örnek eşleşme kontrolleri aynı klasörün analysis bölümündedir.
