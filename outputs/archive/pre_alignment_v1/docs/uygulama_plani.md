# Uygulama ve Deney Planı

Uygulama JupyterLab ve projeye özel Python ortamında yürütülür. Model eğitimi PyTorch ile yerel NVIDIA ekran kartında çalıştırılır; uygunluğu gerçek bir CUDA işlemiyle kontrol edilir. Çalışma altı aşamaya ayrılmıştır.

Güncel durum: gereksinimler, veri hazırlama, üç başlangıçlı ana deneyler, öğrenci göstergeleri, davranış gruplama, gerekçeli öneriler, gerçek kayıtlarla örnek oturum ve küçük metin, ses ve görsel etkileşim örnekleri tamamlandı. Açıklamalı son rapor, ekran görüntüleri ve teslim paketinin tamamlanması sıradadır.

## Birinci aşama Gereksinimler ve ortam

Gereksinim tablosu, kavram sözlüğü, öğrenme blokları, veri giriş ve çıkışları hazırlanır. Ortam sürümleri ve CUDA işlemi kaydedilir. İlk defter veri dosyasının kaynağını ve temel kalite kontrollerini gösterir.

## İkinci aşama Veri hazırlama

Düzeltilmiş ASSIST2009 dosyası ham haliyle saklanır ve SHA256 özeti kaydedilir. Etiket, kimlik, sıralama ve beceri alanları incelenir. Çok becerili bir gerçek olay tek etkileşim olarak tutulur; kavram bilgisi küme olarak temsil edilir. Bu seçim makaledeki çok becerili kayıtları ayırma işleminden farklıysa sonuçlar birebir yeniden üretim olarak adlandırılmaz.

Geçerli cevap ve beceri kayıtları üzerinde en az beş etkileşimi olan öğrenciler tutulur. Öğrenciler sabit bir ayrımla yaklaşık yüzde 70 eğitim, yüzde 15 doğrulama ve yüzde 15 test olarak ayrılır. Zorluk istatistikleri, sayısal dönüşümler ve gruplama yalnızca eğitim kısmına dayanır. Bilinmeyen kimlikler için tanımlı yedek değerler kullanılır ve kapsama oranları raporlanır.

## Üçüncü aşama Model ve karşılaştırma

Önce küçük bir örnek üzerinde veri akışı, hedef cevabın gizlenmesi ve prompt gradyanları kontrol edilir. Model boyutu, dizi uzunluğu ve toplu örnek sayısı GPU belleği ile çalışma süresi ölçülerek seçilir. Bu ön deneme bilimsel sonuç tablosuna eklenmez.

Ana bileşenler LKM, RTM, FBO, PromEnc ve bilgi kavramıyla yönlendirilen promptlardır. Her bağımsız başlangıçta ortak LKM ön eğitilir; promptsuz uyarlama, KPT ve KMPT bu aynı kontrol noktasından başlar. Başlangıçlar 17, 42 ve 2026 olarak kaydedilir. Öğrenci ayrımı bütün yöntemlerde ve başlangıçlarda sabit kalır.

Doğrulama verisi model seçimi ve erken durdurma için kullanılır. Ana sonuçlar ayrılmış testte AUC, tahmin doğruluğu ve ikili çapraz entropiyle raporlanır. Her koşulun tekil sonuçları ile üç başlangıç üzerindeki ortalama ve örnek standart sapması birlikte verilir. Eğitim parametreleri, parametre sayıları ve süreler de saklanır.

## Dördüncü aşama Öğrenci takibi ve öneriler

Referans soru kümeleri eğitimden belirlenir ve dondurulur. Becerilere göre tahmini başarı yüzdesi ile gözlenen doğru cevap oranı ayrı hesaplanır. Yakın dönem davranış özellikleri ölçeklenir; uygun grup sayısı eğitim tarafında siluet ve grup büyüklükleriyle seçilir. Gruplar merkezlerindeki özelliklerle açıklanır.

Çalışma önceliği yakın geçmişte aktif becerilerden referans tahmini en düşük olan beceriye verilir. Öneriler mevcut eğitim sorularının kimlikleri arasından seçilir. Normalde yüzde 65, son üç cevap yanlışsa yüzde 80, son üç cevap doğruysa yüzde 55 doğru cevap olasılığına yakın bir soru hedeflenir. Daha önce cevaplanmamış aday varsa öncelik ona verilir. Bunlar açık seçim kurallarıdır; pedagojik açıdan en iyi eşikler veya öğrenilmiş öneri politikası olarak sunulmaz. Öneriyle birlikte gerekçesi gösterilir.

## Beşinci aşama Örnek oturum ve etkileşimler

Mevcut bir geçmiş, her cevaptan önce tahmin ve her cevaptan sonra durum güncellemesi yapılarak yeniden oynatılır. Sıra ve gösterim zamanı kaydedilir. Küçük metin, ses ve görsel örnekleri bilinen bir beceriyi ifade etme işlevini gösterir. Gerekli kadar örnek içerik hazırlanır; kapsamlı soru bankası veya yeni sorulara model uyarlama yapılmaz.

Çalışan örneklerde ses Whisper base ile, basılı Türkçe yazı Windows OCR ile metne çevrilir. Karakter TF IDF ve lojistik regresyonla öğrenilmiş küçük sınıflandırıcı, kapsam kurallarıyla birlikte mevcut üç toplama ve çıkarma becerisine yönlendirir. Dokuz kontrollü girdinin gerçek çıktıları saklanır. İstekler öğrenci cevap geçmişini değiştirmez. Ayrıntılar [etkileşim örnekleri belgesindedir](metin_ses_gorsel_ornekleri.md).

## Altıncı aşama Rapor ve teslim

Rapor; senaryo, gereksinimler, diyagram, veri sözlüğü, yöntemler, deney düzeni, sonuçlar, örnek oturum ve sınırlılıkları içerir. Her kod ve çıktı bölümü açıklanır, ekran görüntüleri eklenir. Grup üyelerinin bilgileri tamamlanır. Kod, rapor, gerekli küçük örnek dosyalar, ayarlar ve çalıştırma yönergesi tek arşivde birleştirilir. Büyük geçici ortam klasörü ve kişisel çalışma dosyaları teslim paketine alınmaz.
