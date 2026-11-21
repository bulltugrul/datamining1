# Teslimi inceleme kılavuzu

Önce ana raporu, ardından ihtiyaç duyulan kod ve çalışma çıktıları için ayrı eki okuyun. İki PDF de output/pdf klasöründedir. Grup üyelerinin adları ve öğrenci numaraları kapaklarda bulunur.

Güncel uygulamayı JupyterLab içinde 01, 02, 08, 05, 06, 07, 09 sırasıyla inceleyin. Sekizinci defter kayıtlı ağırlıklar ve tahminlerle yöntem denetimini yeniden çalıştırır; bütün eğitimleri yeniden başlatmaz. Üçüncü ve dördüncü defterler bu sohbet içinde geliştirilen ilk sürümün tarihsel sonuçlarıdır. Kullanıcının önceki bağımsız projesinin kodları kullanılmamıştır.

Güncel deney outputs/main/paper_alignment_v2 altındadır. Üç başlangıçta dört koşuldan oluşan 12 yeni eğitim ve aynı sabit ağırlıklarda altı ek dikkat yorumu değerlendirmesi vardır. Ana sonuçlar katı token dikkat yorumunu korur. Öğrenci prototipi, testten önce kaydedilen aynı hedef zaman bloğu yorumunu kullanır; bu, doğrulanmış yazar maskesi iddiası değildir. Etkin seçim configs/active_model.json ile yönlendirilir.

Eğitim, gösterge ve gruplama için veriler; eğitilmiş model ağırlıkları; Whisper ses modeli; gerçek küçük ses ve görsel örnekleri pakete dahildir. Böylece kayıtlı sonuçları incelemek için yeniden eğitim veya ses modelini tekrar indirme gerekmez. Yeni bilgisayarda Python ortamı kurulmalıdır; .venv ve geçici önbellekler pakete alınmaz. OCR örnekleri Windows Türkçe OCR motorunu kullanır. Başlatıcı dosyalarına bu çalışma sırasında müdahale edilmemiştir.

Son bütünleşik kontrolde 59 test geçti; güncel defterler hatasız çalıştı. Dosya kaynakları ve içerik özetleri deney ve denetim klasörlerindedir. Arşiv içindeki TESLIM_DOSYALARI.json her dosyanın boyutunu ve SHA256 özetini listeler. Raporun tarayıcı görüntüleri outputs/report_evidence altında kaynak ve hücrelerle eşleştirilir.

Bu çalışma ölçekli yöntem uygulamasıdır. Makaledeki tam kapasite, dört veri kümesi ve bütün karşılaştırma modelleri yeniden üretilmemiştir. Algoritma 1'deki artırılmış uyarlama verisinin ayrıntıları ile yazarların kesin dikkat maskesi doğrulanamamıştır. Arşiv verisinde gerçek takvim zamanları yoktur; arşiv örneği cevap adımları boyunca gösterilir. Yeni 09 defteri boş bir canlı oturum açar ve yeni değerlendirilmiş cevapları gerçek alım zamanlarıyla kaydeder. İki olay arasında son tahmin tutulur. Paylaşılan yeni örnekler otomatik işlev testidir; insan öğrenme verisi değildir. Önerilerin gerçek öğrenme kazanımı sağladığı ölçülmemiştir. Bu sınırlar raporda açıkça korunur.

Tek ZIP yerel olarak hazırlanmıştır; herhangi bir sisteme yüklenmemiş veya kimseye gönderilmemiştir. Tekrarlanan eski gösterim arşivi ZIP'e alınmaz; tarihsel ana deney ve defter kayıtları korunur.
