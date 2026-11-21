# Öğrenci Takibi Davranış Grupları ve Örnek Oturum

Sistem, eğitilmiş KMPT modeliyle beceri göstergeleri üretir; eğitim geçmişlerinden öğrendiği gruplara güncel öğrenci durumunu atar ve mevcut sorular arasından gerekçeli çalışma önerisi seçer. Örnek oturum gerçek kayıtları sırayla yeniden oynatır. Model ağırlıkları ve referans sorular oturum boyunca sabittir.

Etkin dikkat politikası same_target_step_block. Güncel aynı hedef zaman bloğu yorumu yazarların doğrulanmış maskesi değildir; katı token referansı ve ayrı tanısal maske değerlendirmeleri sekizinci defterde korunur.

## Beceri göstergeleri

Her becerinin başarı göstergesi, önceden eğitim sorularından seçilmiş sabit referans kümesindeki doğru cevap olasılıklarının ortalamasının yüz katıdır. Küme üyeleri ve eşit ağırlıkları oturumda değişmez. Tahmin yalnızca o ana kadar gözlenmiş en fazla 200 etkileşimi kullanır.

Gözlenen doğru cevap oranı ayrı hesaplanır. Grafiklerde son 20 etkileşim içinde ilgili beceriyi içeren cevaplar sayılır; soru sayısı da gösterilir. Tek bir doğru cevap yüzde 100 gözlenen oran üretebilir. Bu, becerinin kesin öğrenildiği anlamına gelmez.

## Davranış gruplama girdileri

Yakın dönem davranış vektöründe dört özellik vardır: son 20 cevaptaki doğru oranı, yakın geçmişte en sık görülen en fazla üç becerinin sabit referans tahmin ortalaması, pozitif cevap sürelerinin medyanının logaritmik dönüşümü ve cevap başına ortalama ipucu kullanımı. Aktif beceri ortalaması gruplama için bağlamsal bir özettir; genel öğrenme yüzdesi olarak sunulmaz.

Süre sınırları yalnızca eğitimden belirlenmiştir: alt sınır 1.409, üst sınır 365.283 saniye. Sıfır ve geçersiz süreler eksik kabul edilir. İpucu sayıları eğitimden belirlenen 5.0 üst sınırında kesilir. Eksikler eğitim medyanlarıyla tamamlanır; ölçekleme eğitimde öğrenilir.

## Grupların öğrenilmesi

2550 eğitim öğrencisinden, öğrenci başına en fazla dört ve en az beş cevap içeren toplam 9598 geçmiş penceresi kullanıldı. Doğrulama ve test öğrencileri grup, ölçek veya sınır öğrenimine alınmadı.

KMeans için k means plus plus başlangıcı, 10 farklı merkez başlangıcı ve sabit 2026 rastgele başlangıcı kullanıldı. İki ile altı grup denendi. Her grubun pencerelerin en az yüzde 5'ini içerdiği adaylar arasından, eğitimden örneklenen en fazla 2000 penceredeki siluet değeri en yüksek olan seçildi.

| Grup sayısı | Siluet | En küçük grup oranı yüzde | Büyüklük koşulu |
|---|---|---|---|
| 2 | 0.4255 | 26.22 | Uygun |
| 3 | 0.3158 | 11.05 | Uygun |
| 4 | 0.2869 | 8.79 | Uygun |
| 5 | 0.2695 | 7.89 | Uygun |
| 6 | 0.2720 | 5.45 | Uygun |

Seçilen grup sayısı 2; siluet değeri 0.4255. Siluet, grupların bu özelliklerdeki ayrışmasını özetler; sınıflandırma doğruluğu veya öğrenme kazanımı yüzdesi değildir.

| Grup | Pencere | En az bir penceresi bu grupta olan öğrenci | Gözlenen doğru oranı yüzde | Referans tahmini yüzde | Tipik cevap süresi saniye | Cevap başına ipucu |
|---|---|---|---|---|---|---|
| 0 | 7081 | 2204 | 75.54 | 71.41 | 23.59 | 0.20 |
| 1 | 2517 | 1059 | 29.42 | 39.65 | 17.79 | 1.35 |

Aynı öğrencinin farklı dönemlerde farklı gruplarda pencereleri bulunabilir; öğrenci sütunu gruplar arasında toplanamaz. Grup numaraları başarı sırası veya kalıcı öğrenme tarzı değildir. Tahmin özelliği eğitim geçmişlerinde aynı modelin göstergelerinden üretildiği için bu gruplama ayrıca bağımsız bir genelleme başarısı ölçümü olarak yorumlanmaz.

## Çalışma önerileri

Öneri, mevcut eğitim sorularından bilinen soru kimlikleri arasından seçilir. Yakın geçmişteki aktif beceriler arasında sabit referans başarısı en düşük olan önceliklendirilir. Daha önce cevaplanmamış aday varsa tekrar sorusu seçilmez. En fazla 256 aday, zorluk aralığını kapsayacak şekilde alınır.

Normal durumda tahmini doğru cevap olasılığı yüzde 65'e yakın bir soru seçilir. Son üç cevap yanlışsa hedef yüzde 80'e çıkarılarak daha erişilebilir; son üç cevap doğruysa yüzde 55'e indirilerek daha zorlayıcı bir çalışma hedeflenir. Bunlar önceden belirlenmiş açık kurallardır; öğrenilmiş öneri politikası veya pedagojik olarak kanıtlanmış en iyi eşikler değildir.

Çıktı mevcut bir soru kimliği, beceri, zorluk, tahmini olasılık ve gerekçedir. Kapsamlı soru bankası hazırlanmaz, yeni yazılmış sorulara model uyarlaması yapılmaz. Davranış grubu öğreticiye ek bağlam sağlar; öğrenciye değişmez bir kişisel özellik atamaz.

## Gerçek kayıtlarla örnek oturum

Örnek, yeterli uzunlukta doğrulama geçmişleri arasından yalnızca kayıt uzunluğu ve sabit rastgele başlangıçla seçildi. İlk 5 cevap başlangıç geçmişidir; sonraki 20 gerçek cevap sırayla işlendi. Öğrenci, sonucun iyi görünmesine veya eğrinin artmasına göre seçilmedi.

Her adımda hedef cevap okunmadan tahmin üretildi. Ardından kayıttaki cevap geçmişe eklendi; referans göstergeleri, gözlenen oranlar, grup ve öneri güncellendi. Yenilenen şey öğrenci geçmişidir; model ağırlıkları çevrimiçi yeniden eğitilmez.

Bu örnekte gruplar arasında 4 geçiş görüldü. Göstergeler artabilir, azalabilir veya sabit kalabilir. Çizgiler ayrık cevap adımlarını birleştirir; aralarda gerçekleşmiş sürekli öğrenme ölçümü değildir.

### Oturum sonundaki göstergeler

| Beceri | Sabit referans tahmini yüzde | Son pencerede gözlenen yüzde | Bu orana dayanak cevap sayısı | Referans soru sayısı |
|---|---|---|---|---|
| Absolute Value | 58.78 | 100.00 | 4 | 32 |
| Beceri 74 | 39.14 | 100.00 | 1 | 32 |
| Addition Whole Numbers | 53.24 | 100.00 | 1 | 32 |
| Addition and Subtraction Integers | 43.92 | 28.57 | 7 | 32 |
| Addition and Subtraction Positive Decimals | 32.61 | 100.00 | 1 | 32 |
| Addition and Subtraction Fractions | 22.74 | 0.00 | 7 | 32 |

Kaynakta tekil adı güvenle eşlenemeyen ek kavramlar beceri kodlarıyla gösterilir. Yakın dönem oranının yükselmesi, eski yanlış cevapların pencere dışına çıkmasından da kaynaklanabilir. Tahmini gösterge ile gözlenen oran aynı ölçü değildir.

### Öneri gerekçesi örnekleri

Başlangıç: Beceri 74: izlenen beceriler arasında referans başarı tahmini en düşük olan beceri; sabit referans başarı tahmini %38.8. orta düzeyde zorlayıcı başarı hedefi. Seçilen soruda tahmin %64.6, hedef %65.

Son adım: Addition and Subtraction Fractions: izlenen beceriler arasında referans başarı tahmini en düşük olan beceri; sabit referans başarı tahmini %22.7. orta düzeyde zorlayıcı başarı hedefi. Seçilen soruda tahmin %60.0, hedef %65.

Yeniden oynatımda bir sonraki gerçek kayıt, önerilen sorudan farklı olabilir. Bu akış önerinin uygulandığı veya öğrenmeyi artırdığı bir müdahale deneyi değildir.

## Zaman bilgisi

Gösterim başlangıcı 2026-11-15T18:27:14.932+03:00; son adımın gösterim saati 2026-11-15T18:27:16.124+03:00. Bunlar bu program çalıştırmasının gerçek saatleridir. Kaynakta geçmiş olayların takvim tarihleri bulunmadığı için kaynak zamanları boş bırakılmıştır. Grafiklerin birincil ekseni etkileşim adımıdır; geçmiş öğrencinin günlere yayılmış çalışma veya unutma süresi üretilmez.

## Güvenilirlik denetimi

9598 eğitim penceresinin özellikleri yeniden hesaplandı; grup seçimi, ölçeklerin eğitim istatistikleri, gerçek cevap eşleşmeleri ve öneri kuralları doğrulandı. Bilgi takibi ağırlıkları, grup modeli ve referans kümesi oturumda değişmedi. Önceki ana deneyin model kaynakları ve sonuçları korunuyor.

Kayıtlar outputs/patterns, outputs/replay ve outputs/validation/student_tracking_integrity.json altında bulunur. Güncel yöntem seçimi oturum kaydının model_selection alanındadır. Düzeltilmiş yöntem için [yeni deney sonuçları](duzeltilmis_yontem_sonuclari.md), önceki sürüm için [ilk deney sonuçları](ana_deney_sonuclari.md) ayrı saklanır.

## Yöntem kaynakları

Gruplama ve siluet uygulaması için [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [siluet ölçüsü](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) ve [eksik değer tamamlama](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) belgeleri esas alınmıştır.
