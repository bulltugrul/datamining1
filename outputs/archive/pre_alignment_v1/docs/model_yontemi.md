# Bilgi Takibi Modeli ve Eğitim Yöntemi

Uygulama; LKM ön eğitimi, promptsuz uyarlama, KPT ve KMPT bileşenlerini içerir. İlk küçük deneme bu bileşenlerin çalışmasını ve veri güvenilirliğini doğruladı. Bağımsız test verisindeki üç başlangıçlı ana karşılaştırma tamamlandı; [ana deney sonuçları](ana_deney_sonuclari.md) ayrı belgede sunuluyor.

## Etkileşim temsili

Bir geçmiş olayı soru kimliği, bir veya daha fazla beceri, doğru veya yanlış cevap ve eğitimden hesaplanmış soru zorluğuyla temsil edilir. Soru, beceri ve cevap temsilleri ile zorluk ve konum temsilleri birleştirilir. Bir olay birden fazla beceri içeriyorsa geçerli beceri temsillerinin ortalaması kullanılır.

Soru için PAD, UNK ve MASK; beceri için PAD ve UNK ayrılmış kimliklerdir. Eğitimde görülmeyen soru veya becerinin temsili başlangıçta sıfırdır. Dolgu konumları dikkat hesabında anahtar olarak kullanılamaz.

## LKM ön eğitimi

Ön eğitim yalnızca eğitim öğrencilerinin kayıtlarını kullanır. Geçmişler en fazla 128 olaylık parçalara bölünür. En az iki olay içeren her pencere iki ardışık parçaya ayrılır; başına CLS ve parçalar arasına SEP eklenir.

RTM için olay konumlarının yaklaşık yüzde 15'i seçilir. Kısa bir dizide en az bir konum seçilir. Seçilen konumdaki soru ve cevap bileşenleri ayrı ayrı yüzde 80 olasılıkla maskelenir, yüzde 10 olasılıkla aynı dizideki başka bir gerçek konumdan alınır, yüzde 10 olasılıkla değiştirilmez. Kayıp yalnızca seçilen konumlarda hesaplanır. Beceri ve zorluk yardımcı bilgi olarak korunur.

FBO olumlu örneğinde iki parça aynı özgün pencereden gelir. Olumsuz örnekte ikinci parça farklı bir eğitim öğrencisinin penceresinden alınır. Olumlu ile olumsuz örnek oranı, yuvarlama farkı dışında 2:1'dir. Ön eğitimde iki yönlü dikkat kullanılır.

Toplam kayıp, maskeli soruların çok sınıflı çapraz entropisi, maskeli cevapların ikili çapraz entropisi ve FBO ikili çapraz entropisinin toplamıdır. Her bileşen kendi geçerli örnekleri üzerinde ortalanır. Üç kayıp ayrı kaydedilir; soru başlığının kaybı cevap tahmini BCE değeriyle aynı ölçü olarak yorumlanmaz.

## Üç uyarlama yöntemi

Her başlangıç için aynı ön eğitimli LKM ağırlıkları üç yönteme kopyalanır. Ortak başlangıç, ağırlıkların içerik özetiyle doğrulanır.

| Yöntem | Tahmin yapısı | Eğitim hedefi |
|---|---|---|
| Promptsuz uyarlama | Maskeli cevap içeren hedef etkileşim ve yeni sınıflandırma başlığı | Sonraki cevabın ikili kaybı |
| KPT | Öğrenilebilir promptlar ve bilgi kavramıyla yönlendirilmiş şablon; ön eğitimli cevap başlığı | Sonraki cevabın ikili kaybı |
| KMPT | KPT ile aynı cevap şablonu ve yardımcı soru şablonu | Örneklerin yaklaşık yüzde 80'inde cevap, yüzde 20'sinde soru tahmini |

Üç yöntemde de LKM uyarlanır. KPT ve KMPT'de PromEnc ile soft promptlar da eğitilir. FBO başlığı uyarlamada dondurulur. Soru başlığı yalnızca KMPT'de; ön eğitimli cevap başlığı KPT ve KMPT'de güncellenir. Promptsuz yöntemde yeni sınıflandırma başlığı eğitilir.

Bu deney saf parametre verimli prompt eğitimi karşılaştırması değildir. KPT ve KMPT'nin farkları şablon, ek kodlayıcı, başlık kullanımı ve hedeflerle birlikte değerlendirilir. Eğitilen parametre sayıları ayrıca raporlanır; farklar yalnızca soft promptlara atfedilmez.

## Tahmin sırasında bilgi sınırı

Bir tahmin örneği hedef olaydan önceki en fazla 128 etkileşimi içerir. Hedef cevabı etiket olarak saklanır; tahmin girdisine alınmaz. Sonraki cevabı tahmin eden LKM dikkat maskesi nedenseldir.

KPT şablonunun sırası soft promptlar, hedef soru, beceri ve maskeli cevaptır. KMPT yardımcı şablonunda bilinen cevap ve beceri önce, maskeli soru sonradır. Maskeli bileşenin sonda bulunması, nedensel LKM'nin promptları görebilmesini sağlar. Yardımcı şablon yalnızca KMPT eğitimi sırasında kullanılabilir; değerlendirme yolu her zaman cevap maskeli şablondur.

PromEnc yalnızca bu şablonun bileşenleri üzerinde iki yönlü çalışır. Tahminde şablonun içinde hedefin gerçek cevabı veya sonraki olaylar bulunmaz. LKM'nin geçmiş etkileşimleri nedensel kalır. Bu düzen, makaledeki şablon sırası ve tahminde PromEnc dikkatini değiştirme tarifinin birebir kopyası değildir; fark açıkça belirtilir.

Konum kimlikleri öğrencinin gerçek geçmiş uzunluğuna göre atanır. Aynı öğrenci başka uzunlukta geçmişlerle aynı toplu girdiye girdiğinde dolgu sayısının tahmini değiştirmemesi denetlenir.

## Kapasite ve donanım

Prototip LKM iki Transformer katmanı, dört dikkat başlığı ve 128 boyutlu temsil kullanır. PromEnc bir katman ve sekiz öğrenilebilir prompt içerir. Bu kapasite makalenin A100 üzerinde kullandığı modelden küçüktür; sonuçlar birebir makale yeniden üretimi olarak sunulmaz.

CUDA üzerinde destek varsa bfloat16 karma hassasiyet kullanılır; model ağırlıkları standart hassasiyette kalır. Küçük deneme 128'lik toplu girdilerle çalışmıştır. Yalnızca eğitim girdileriyle yapılan hız ve bellek ölçümünde ana deney için 512'lik toplu girdi seçilmiştir. 1024'teki küçük hız kazanımı yaklaşık iki kat bellek gerektirdiğinden 512 tercih edilir.

## Seçim ve değerlendirme

En iyi uyarlama ağırlıkları doğrulama BCE değerinin en düşük olduğu dönemden seçilir. AUC ve doğruluk o aynı dönemin değerleridir; her ölçü için farklı dönem seçilmez. KMPT'nin yardımcı soru kaybı ortak model karşılaştırma ölçüsü değildir.

Ana deney başlangıçları 17, 42 ve 2026'dır. Öğrenci ayrımı sabit kalır. Ayarlar ana deneyde test sonuçlarına göre değiştirilmez. Her yöntemin seçilmiş ağırlıkları bağımsız testte değerlendirilir; eğitimde görülen soru alt grubunun sonucu da ayrıca saklanır.

## Kaynaklar

- Zhu, Qiu ve Zhou, [Prompt tuning for improving knowledge tracing in intelligent tutoring](https://link.springer.com/article/10.1007/s10618-026-01208-9), özellikle yöntem bölümleri 3.2 ile 3.5. Kullanıcının sağladığı tam metin esas alınmıştır.
- [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html): katmanların ayrı başlatılması ve dikkat maskeleri.
- [PyTorch karma hassasiyet](https://docs.pytorch.org/docs/2.14/amp.html): CUDA autocast ve gerektiğinde gradyan ölçekleme.
