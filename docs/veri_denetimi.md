# ASSIST2009 Veri Denetimi

Düzeltilmiş collapsed Skill Builder dosyası yeniden incelendi. Ham dosyada 346.860 satır, 4.217 öğrenci, 26.688 soru ve 123 beceri bulunuyor. Kaynak dosya önceki proje kodlarından bağımsız olarak işlendi; ham kopyanın SHA256 özeti saklandı.

## Uygunluk koşulları

Beceri bilgisi eksik veya geçersiz 63.755 satır çıkarılıyor. Kalan olaylar içinde en az beş etkileşimi olan öğrenciler tutulduğunda 3.644 öğrenci, 17.727 soru ve 123 beceriye ait 281.890 olay kalıyor. Çok becerili 47.037 ham satırdaki her gerçek cevap tek olay olarak korunuyor. En fazla dört becerili olay bulunuyor. Çelişkili cevap içeren kopya olay saptanmadı.

## Öğrenci grupları

Ayrım başlangıcı 2026. Bir öğrencinin bütün kayıtları tek grupta kalıyor.

| Grup | Öğrenci | Etkileşim | Eğitimde görülmeyen soru içeren etkileşim |
|---|---|---|---|
| Eğitim | 2550 | 197213 | 0 |
| Doğrulama | 546 | 39338 | 548 |
| Test | 548 | 45339 | 308 |

Doğrulama ve testte eğitimde görülmeyen beceri içeren olay bulunmuyor. Eğitimde görülmeyen sorular elenmeden UNK kimliğiyle korunuyor. Bilinen beceriler olayın sağlanan girdisidir; bilinmeyen sorunun zorluğu genel eğitim hata oranına eşit atanıyor. Bu durum yeni yazılmış soru bankası hazırlama veya ona uyarlama kapsamına girmez.

## Eğitimden hesaplanan özellikler

Genel eğitim hata oranı yaklaşık yüzde 33,94. Bilinen her sorunun zorluğu, yalnızca eğitim öğrencilerinin o sorudaki hata oranından hesaplanıyor. Soru ve beceri sözlükleri eğitimde oluşturuluyor. Sabit referans soruları da eğitim sorularından seçiliyor; hiçbir seçim test cevaplarına dayanmıyor.

## Zaman bilgisi

order_id ve position sıralama sağlıyor. ms_first_response ve overlap_time süre alanlarıdır; geçmiş olayın tarihini vermez. Örnek oturum bu kayıtları sırayla yeniden oynatacak ve gösterim saatini ayrı kaydedecek. Veriden günler veya haftalar ölçeğinde gerçek öğrenme kazanımı çıkarılmayacak.

## Doğrulama

Otomatik kontroller öğrenci gruplarının kesişmediğini, doğrulama ve test cevapları değiştirildiğinde eğitimden hesaplanan zorlukların ve referans sorularının değişmediğini, çok becerili kopyaların tek olay olarak tutulduğunu ve çelişkili olayların çıkarıldığını doğruluyor. Bu aşamanın sayıları veri hazırlama sonuçlarıdır; model başarısı ölçümü değildir.
