# Gerçek zamanlı oturum kullanımı

09_Gercek_Zamanli_Oturum_Takibi.ipynb defteri, yeni değerlendirilmiş cevapları gerçek alım saatleriyle kaydeder ve beceri tahminlerini zaman ekseninde gösterir. Yeni katılımcı geçmişi boştur; ASSIST2009 öğrencisinin kayıtları bu katılımcıya aktarılmaz. Eğitim, doğrulama ve test verileri değiştirilmez; model yeniden eğitilmez.

## Kullanım

Defteri JupyterLab içinde baştan sona çalıştırın. Alt bölümdeki ekranda bir katılımcı etiketiyle yeni oturum açın. Beceri ve mevcut soru kimliğini seçerek **Soruyu başlat** düğmesine basın. Soruyu dışarıda sunup değerlendirdikten sonra doğru veya yanlış sonucu ve varsa ipucu sayısını kaydedin. Kaynakta soru metni bulunmadığı için bu ekran soruyu kimliğiyle gösterir; yeni soru bankası veya otomatik cevap değerlendirme içermez.

Her cevap öncesinde tahmin yapılır. Bekleyen soru geçmişe eklenmez. Sonuç alınınca olay ve gerçek saatleri kaydedilir; göstergeler, davranış grubu ve öneri güncellenir. Başlangıçta gözlenen doğru oranı yoktur. Davranış grubu en az beş yeni cevaptan sonra gösterilir.

İlk cevap öncesindeki model skoru bir başlangıç öncülüdür; öğrenciye ait öğrenme gözlemi içermez. Sıfır geçmiş durumu ana test düzeninden farklıdır ve bu ilk skor için ayrıca doğrulanmış bir soğuk başlangıç başarısı iddiası yoktur.

Oturum kimliğini saklayın. Oturumu kapatıp defteri yeniden açtığınızda kimliği girerek **Kaydı aç** düğmesiyle devam edebilirsiniz. Kapatılmış kayıt inceleme için açılır; yeni cevap için yeni oturum gerekir. Aynı oturumu tek açık çalışma görünümünde yönetin. Aynı cevabın tekrar gönderilmesi ikinci olay oluşturmaz; eski bir görünüm güncel dosyanın üstüne yazamaz.

## Kayıt alanları

| Alan | Anlam |
|---|---|
| session_id / participant_alias | Oturum kimliği ve katılımcı etiketi. |
| question_token / original_problem_id | Bekleyen sorunun tekil gönderim kimliği ve kaynak soru kimliği. |
| predicted_at / presented_at | Hedef cevaptan önce tahminin kaydedildiği ve sorunun başlatıldığı saat. |
| answer_received_at | Değerlendirilmiş sonucun sunucuya ulaştığı gerçek UTC saati. |
| response_duration_seconds | Aynı süreçte monoton saat farkı; yeniden açılan bekleyen soruda UTC farkı. |
| state_updated_at / after | Cevap sonrası durumun üretildiği saat ve beceri, davranış, öneri çıktıları. |
| response_origin | manual_assessed_response veya automated_smoke_test. |
| record_sha256 / event_sha256 | Kazara kayıt değişikliğini tespit eden içerik özetleri; kimlik doğrulama imzası değildir. |

Kalıcı dosyalar outputs/live/sessions altında oturum kimliğiyle saklanır. Her işlem geçici dosyaya yazılıp atomik olarak kaydedilir. Saat dilimsiz veya geriye giden saat kayıtları reddedilir. Arayüz, saatleri Europe/Istanbul bölgesinde gösterir.

## Zaman grafiğinin yorumu

Yatay eksen gerçek kayıt zamanıdır. Model yalnızca yeni değerlendirilmiş cevap geldiğinde yeniden sorgulanır; iki olay arasında son değer tutulur. Çizgi aralıkları sürekli öğrenme ölçümü veya ara değer tahmini değildir. Geçen süre KT modelinin girdisi yapılmamış, süreye bağlı unutma modeli eğitilmemiştir.

outputs/live/demo_session.json gerçek duvar saatiyle çalışan altı cevaplık otomatik API işlev testidir. Arayüz kontrolü de ayrı bir otomatik etiketli oturumdur. Önceden belirlenen doğru/yanlış sonuçları yeni saatlere kaydetmek, bu sonuçları gerçek öğrenci cevabı veya öğrenme kazanımı verisi yapmaz. Gerçek katılımcıyla uzun süreli öğrenme ve öneri etkisi çalışması ayrıca gereklidir.
