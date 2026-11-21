# Yöntem Düzeltmesi Sonrası Deney Sonuçları

Makaledeki şablon sırası ve eğitim/tahmin dikkat ayrımı, FBO örnekleri, RTM/FBO kayıp ölçeği ve BERT tarzı katman düzeni düzeltilerek yeni ön eğitimden başlandı. Üç başlangıçta aynı ortak ön eğitimden promptsuz, KPT ve KMPT çalıştırıldı. Dördüncü koşul yalnızca KMPT şablon yerleşimini değiştirir; bu koşul makale tarifi olarak adlandırılmaz.

| Yöntem | Test AUC | Test doğruluğu yüzde | Test BCE |
|---|---|---|---|
| Promptsuz | 0.7585 ± 0.0032 | 72.59 ± 0.31 | 0.5788 ± 0.0051 |
| KPT metin tarifi | 0.7489 ± 0.0010 | 71.47 ± 0.77 | 0.5596 ± 0.0098 |
| KMPT metin tarifi | 0.7522 ± 0.0043 | 72.27 ± 0.29 | 0.5522 ± 0.0035 |
| KMPT prefix kontrolü | 0.7569 ± 0.0027 | 72.47 ± 0.26 | 0.5460 ± 0.0036 |

Standart sapma tek öğrenci ayrımında üç başlangıç arasındadır; güven aralığı veya anlamlılık testi değildir.

## Her koşulun doğrulama seçimi

| Başlangıç | Koşul | Seçilen dönem | Doğrulama BCE | Test AUC |
|---|---|---|---|---|
| 17 | Promptsuz | 1 | 0.5932 | 0.7595 |
| 17 | KPT metin tarifi | 1 | 0.5787 | 0.7500 |
| 17 | KMPT metin tarifi | 1 | 0.5540 | 0.7524 |
| 17 | KMPT prefix kontrolü | 2 | 0.5489 | 0.7597 |
| 42 | Promptsuz | 1 | 0.5830 | 0.7612 |
| 42 | KPT metin tarifi | 1 | 0.5591 | 0.7486 |
| 42 | KMPT metin tarifi | 1 | 0.5595 | 0.7478 |
| 42 | KMPT prefix kontrolü | 2 | 0.5575 | 0.7566 |
| 2026 | Promptsuz | 1 | 0.5872 | 0.7549 |
| 2026 | KPT metin tarifi | 1 | 0.5567 | 0.7482 |
| 2026 | KMPT metin tarifi | 2 | 0.5611 | 0.7564 |
| 2026 | KMPT prefix kontrolü | 1 | 0.5530 | 0.7543 |

## Parametre ve eğitim kapsamı

| Koşul | Toplam parametre | Eğitilen parametre |
|---|---|---|
| Promptsuz | 4845489 | 2643201 |
| KPT metin tarifi | 5029808 | 2827649 |
| KMPT metin tarifi | 5029808 | 5029679 |
| KMPT prefix kontrolü | 5029808 | 5029679 |

LKM, ilgili ön eğitimli tahmin başlıkları ve prompt kodlayıcı birlikte güncellenir; bu saf PEFT deneyi değildir. Prefix kontrolü aynı LKM ve prompt başlangıçlarından başlar; ortak dönemlerde aynı 80/20 örnek seçimleri kullanıldığı ayrıca denetlenir.

## Prompt duyarlılığı

Ayrı 16 doğrulama geçmişinde eğitim kipinde prompt gradyanı, tahmin kipinde promptlara sabit büyüklükte rastgele değişiklikten sonra olasılık farkı ölçüldü. Model ağırlıkları veya kaydedilmiş kontrol noktaları bu tanısal işlemle değiştirilmedi. Katı nedensellikte sondaki promptların doğrudan etkisi yoktur; baştaki prompt kontrolü bu erişim yolunu açar. Bu bulgu makalenin metinsel tarifine ilişkindir; yazarların kodunun incelendiği iddiası değildir.

| Başlangıç | Koşul | Eğitim prompt gradyanı L1 | Tahminde en büyük olasılık değişimi |
|---|---|---|---|
| 17 | kpt | 7.71587 | 0 |
| 17 | kmpt | 7.0682 | 0 |
| 17 | kmpt_prefix | 8.25541 | 0.263285 |
| 42 | kpt | 8.21053 | 0 |
| 42 | kmpt | 8.03604 | 0 |
| 42 | kmpt_prefix | 6.83637 | 0.075992 |
| 2026 | kpt | 3.99807 | 0 |
| 2026 | kmpt | 10.7048 | 0 |
| 2026 | kmpt_prefix | 8.88885 | 0.114123 |

## Kullanılan örnek model ve sınırlar

Önceden belirlenen KMPT metin tarifi içinde yalnızca doğrulama BCE değeriyle başlangıç 17 seçildi. Seçim testte en yüksek AUC veren koşulu seçme işlemi değildir. Örnek oturum ve sonraki öğrenci göstergeleri bu ağırlıklara bağlanır.

Toplam 12 koşul aynı 44791 test hedefinde değerlendirildi. Tüm eğitimler bitip ayarlar ve kontrol noktaları kilitlendikten sonra test hesaplandı. Bununla birlikte bu test ayrımı eski sürümde zaten görülmüştür; yeni sayılar tamamen yeni ve dokunulmamış doğrulayıcı deney olarak sunulmaz.

Ölçekli model kapasitesi, öğrenci bazlı 70/15/15 ayrımı, çok kavramlı tek olayların korunması, yalnızca eğitimden zorluk hesabı ve sonlu eğitim bütçesi makalenin deney düzeninden ayrılır. Bunlar makale düzeyi sonuçların birebir yeniden üretildiği iddiasını engeller. Promptsuz kontrol yeni bir MLP başlığı ve zorluk dahil tam hedef etkileşimi kullanır; KPT ailesi makaledeki e,c,r şablonunu ve ön eğitimli başlıkları kullanır. Dolayısıyla ana üçlü bütün yöntem paketlerini karşılaştırır; tek başına prompt eklemenin nedensel etkisini ayırmaz. Eski ve yeni sürüm de birden fazla bileşende değiştiğinden farklar tek bir düzeltmeye atfedilmez. Gerekçeli denetim [yöntem uyumu belgesindedir](makale_yontem_uyumu.md).

Toplam yeni deney süresi 109.17 dakika; en yüksek ayrılan GPU belleği 1687.5 MiB. Sonuçlar outputs/main/paper_alignment_v2 altında, ham tahminlerden yeniden hesaplanan denetim analysis/integrity.json içindedir.

## Aynı hedef zaman bloğu yorumu (ayrı tanısal kontrol)

Şablon sırası korunur; hedefin bilinen veya MASK tokenları aynı zaman adımında iki yönlü erişir. Geçmiş konumları sonraki etkileşimleri göremez. Bu, doğrulanmış yazar maskesi değildir. Aynı kilitli ağırlıklarda altı ek tahmin değerlendirmesi yapıldı; yeni eğitim veya bu maske için dönem seçimi yapılmadı. On iki ana eğitim sonucu yukarıda katı token yorumu olarak korunur.

|Yöntem|Blok AUC ortalama|Blok doğruluk|Blok BCE|
|---|---:|---:|---:|
|kpt|0.7529|71.12%|0.5603|
|kmpt|0.7530|72.24%|0.5556|

Çalışan prototip için blok politikası testten önce kaydedildi; KMPT başlangıcı yalnızca bu politikadaki doğrulama BCE değeriyle seçildi: 17. Seçilen modelde pseudo token değişikliğinin en büyük olasılık etkisi 0.229969. Güncel seçim interpretations/prototype_selection.json dosyasındadır; yukarıdaki analysis seçimi katı yorumun referans kaydıdır.
