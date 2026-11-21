# Öğrenci Takibi Davranış Grupları ve Örnek Oturum

Sistem, eğitilmiş KMPT modeliyle beceri göstergeleri üretir; eğitim geçmişlerinden öğrendiği gruplara güncel öğrenci durumunu atar ve mevcut sorular arasından gerekçeli çalışma önerisi seçer. Örnek oturum gerçek kayıtları sırayla yeniden oynatır. Model ağırlıkları ve referans sorular oturum boyunca sabittir.

## Beceri göstergeleri

Her becerinin başarı göstergesi, önceden eğitim sorularından seçilmiş sabit referans kümesindeki doğru cevap olasılıklarının ortalamasının yüz katıdır. Küme üyeleri ve eşit ağırlıkları oturumda değişmez. Tahmin yalnızca o ana kadar gözlenmiş en fazla 128 etkileşimi kullanır.

Gözlenen doğru cevap oranı ayrı hesaplanır. Grafiklerde son 20 etkileşim içinde ilgili beceriyi içeren cevaplar sayılır; soru sayısı da gösterilir. Tek bir doğru cevap yüzde 100 gözlenen oran üretebilir. Bu, becerinin kesin öğrenildiği anlamına gelmez.

## Davranış gruplama girdileri

Yakın dönem davranış vektöründe dört özellik vardır: son 20 cevaptaki doğru oranı, yakın geçmişte en sık görülen en fazla üç becerinin sabit referans tahmin ortalaması, pozitif cevap sürelerinin medyanının logaritmik dönüşümü ve cevap başına ortalama ipucu kullanımı. Aktif beceri ortalaması gruplama için bağlamsal bir özettir; genel öğrenme yüzdesi olarak sunulmaz.

Süre sınırları yalnızca eğitimden belirlenmiştir: alt sınır 1.409, üst sınır 365.283 saniye. Sıfır ve geçersiz süreler eksik kabul edilir. İpucu sayıları eğitimden belirlenen 5.0 üst sınırında kesilir. Eksikler eğitim medyanlarıyla tamamlanır; ölçekleme eğitimde öğrenilir.

## Grupların öğrenilmesi

2550 eğitim öğrencisinden, öğrenci başına en fazla dört ve en az beş cevap içeren toplam 9598 geçmiş penceresi kullanıldı. Doğrulama ve test öğrencileri grup, ölçek veya sınır öğrenimine alınmadı.

KMeans için k means plus plus başlangıcı, 10 farklı merkez başlangıcı ve sabit 2026 rastgele başlangıcı kullanıldı. İki ile altı grup denendi. Her grubun pencerelerin en az yüzde 5'ini içerdiği adaylar arasından, eğitimden örneklenen en fazla 2000 penceredeki siluet değeri en yüksek olan seçildi.

| Grup sayısı | Siluet | En küçük grup oranı yüzde | Büyüklük koşulu |
|---|---|---|---|
| 2 | 0.4647 | 23.18 | Uygun |
| 3 | 0.3115 | 10.75 | Uygun |
| 4 | 0.2908 | 8.12 | Uygun |
| 5 | 0.2784 | 7.66 | Uygun |
| 6 | 0.2830 | 5.15 | Uygun |

Seçilen grup sayısı 2; siluet değeri 0.4647. Siluet, grupların bu özelliklerdeki ayrışmasını özetler; sınıflandırma doğruluğu veya öğrenme kazanımı yüzdesi değildir.

| Grup | Pencere | En az bir penceresi bu grupta olan öğrenci | Gözlenen doğru oranı yüzde | Referans tahmini yüzde | Tipik cevap süresi saniye | Cevap başına ipucu |
|---|---|---|---|---|---|---|
| 0 | 7373 | 2271 | 74.48 | 76.12 | 24.06 | 0.22 |
| 1 | 2225 | 955 | 26.89 | 45.83 | 16.24 | 1.44 |

Aynı öğrencinin farklı dönemlerde farklı gruplarda pencereleri bulunabilir; öğrenci sütunu gruplar arasında toplanamaz. Grup numaraları başarı sırası veya kalıcı öğrenme tarzı değildir. Tahmin özelliği eğitim geçmişlerinde aynı modelin göstergelerinden üretildiği için bu gruplama ayrıca bağımsız bir genelleme başarısı ölçümü olarak yorumlanmaz.

## Çalışma önerileri

Öneri, mevcut eğitim sorularından bilinen soru kimlikleri arasından seçilir. Yakın geçmişteki aktif beceriler arasında sabit referans başarısı en düşük olan önceliklendirilir. Daha önce cevaplanmamış aday varsa tekrar sorusu seçilmez. En fazla 256 aday, zorluk aralığını kapsayacak şekilde alınır.

Normal durumda tahmini doğru cevap olasılığı yüzde 65'e yakın bir soru seçilir. Son üç cevap yanlışsa hedef yüzde 80'e çıkarılarak daha erişilebilir; son üç cevap doğruysa yüzde 55'e indirilerek daha zorlayıcı bir çalışma hedeflenir. Bunlar önceden belirlenmiş açık kurallardır; öğrenilmiş öneri politikası veya pedagojik olarak kanıtlanmış en iyi eşikler değildir.

Çıktı mevcut bir soru kimliği, beceri, zorluk, tahmini olasılık ve gerekçedir. Kapsamlı soru bankası hazırlanmaz, yeni yazılmış sorulara model uyarlaması yapılmaz. Davranış grubu öğreticiye ek bağlam sağlar; öğrenciye değişmez bir kişisel özellik atamaz.

## Gerçek kayıtlarla örnek oturum

Örnek, yeterli uzunlukta doğrulama geçmişleri arasından yalnızca kayıt uzunluğu ve sabit rastgele başlangıçla seçildi. İlk 5 cevap başlangıç geçmişidir; sonraki 20 gerçek cevap sırayla işlendi. Öğrenci, sonucun iyi görünmesine veya eğrinin artmasına göre seçilmedi.

Her adımda hedef cevap okunmadan tahmin üretildi. Ardından kayıttaki cevap geçmişe eklendi; referans göstergeleri, gözlenen oranlar, grup ve öneri güncellendi. Yenilenen şey öğrenci geçmişidir; model ağırlıkları çevrimiçi yeniden eğitilmez.

Bu örnekte gruplar arasında 6 geçiş görüldü. Göstergeler artabilir, azalabilir veya sabit kalabilir. Çizgiler ayrık cevap adımlarını birleştirir; aralarda gerçekleşmiş sürekli öğrenme ölçümü değildir.

### Oturum sonundaki göstergeler

| Beceri | Sabit referans tahmini yüzde | Son pencerede gözlenen yüzde | Bu orana dayanak cevap sayısı | Referans soru sayısı |
|---|---|---|---|---|
| Absolute Value | 64.02 | 100.00 | 4 | 32 |
| Beceri 74 | 50.36 | 100.00 | 1 | 32 |
| Addition Whole Numbers | 55.97 | 100.00 | 1 | 32 |
| Addition and Subtraction Integers | 51.64 | 28.57 | 7 | 32 |
| Addition and Subtraction Positive Decimals | 52.54 | 100.00 | 1 | 32 |
| Addition and Subtraction Fractions | 57.64 | 0.00 | 7 | 32 |

Kaynakta tekil adı güvenle eşlenemeyen ek kavramlar beceri kodlarıyla gösterilir. Yakın dönem oranının yükselmesi, eski yanlış cevapların pencere dışına çıkmasından da kaynaklanabilir. Tahmini gösterge ile gözlenen oran aynı ölçü değildir.

### Öneri gerekçesi örnekleri

Başlangıç: Beceri 74: izlenen beceriler arasında referans başarı tahmini en düşük olan beceri; sabit referans başarı tahmini %54.7. orta düzeyde zorlayıcı başarı hedefi. Seçilen soruda tahmin %64.8, hedef %65.

Son adım: Addition and Subtraction Integers: izlenen beceriler arasında referans başarı tahmini en düşük olan beceri; sabit referans başarı tahmini %51.6. orta düzeyde zorlayıcı başarı hedefi. Seçilen soruda tahmin %64.8, hedef %65.

Yeniden oynatımda bir sonraki gerçek kayıt, önerilen sorudan farklı olabilir. Bu akış önerinin uygulandığı veya öğrenmeyi artırdığı bir müdahale deneyi değildir.

## Zaman bilgisi

Gösterim başlangıcı 2026-11-15T14:17:37.839+03:00; son adımın gösterim saati 2026-11-15T14:17:38.764+03:00. Bunlar bu program çalıştırmasının gerçek saatleridir. Kaynakta geçmiş olayların takvim tarihleri bulunmadığı için kaynak zamanları boş bırakılmıştır. Grafiklerin birincil ekseni etkileşim adımıdır; geçmiş öğrencinin günlere yayılmış çalışma veya unutma süresi üretilmez.

## Güvenilirlik denetimi

9598 eğitim penceresinin özellikleri yeniden hesaplandı; grup seçimi, ölçeklerin eğitim istatistikleri, gerçek cevap eşleşmeleri ve öneri kuralları doğrulandı. Bilgi takibi ağırlıkları, grup modeli ve referans kümesi oturumda değişmedi. Önceki ana deneyin model kaynakları ve sonuçları korunuyor.

Kayıtlar outputs/patterns, outputs/replay ve outputs/validation/student_tracking_integrity.json altında bulunur. Model performansının bağımsız test sonuçları [ana deney belgesindedir](ana_deney_sonuclari.md).

## Yöntem kaynakları

Gruplama ve siluet uygulaması için [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [siluet ölçüsü](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) ve [eksik değer tamamlama](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) belgeleri esas alınmıştır.
