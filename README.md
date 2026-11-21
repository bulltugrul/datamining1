# Prompt Tuning ile Bilgi Takibi

ASSIST2009 verisi üzerinde öğrencinin sonraki cevabını tahmin eden, beceri bazında gelişimi izleyen ve çalışma önerileri üreten akıllı öğretim prototipi.

Güncel yöntem, makaleyle eşitlik ve dikkat düzeyinde denetlenmiş ikinci sürümdür. Ana sonuçlar `notebooks/08_Makale_Yontem_Denetimi_ve_Duzeltilmis_Deneyler.ipynb` içindedir. Öğrenci takibi ve etkileşimler için 05–07 defterleri yeni ağırlıklarla yeniden çalıştırılır; 03–04 defterleri ilk sürümün tarihsel kaydıdır. Bu çalışma ölçekli yöntem uygulamasıdır; makaledeki bütün deneylerin birebir tekrarı olarak sunulmaz.

## Çalışma ortamı

Ana uygulama **JupyterLab**. Çalışma defterleri tarayıcıda açılır; Python kodu ve model eğitimi yerel bilgisayarda çalışır. Eğitim için PyTorch ve uygun olduğunda NVIDIA CUDA kullanılır. VS Code, yardımcı Python dosyalarını düzenlemek için kullanılabilir.

`start_jupyter.cmd` dosyasını çift tıklayarak çalışma defterini açın. İlk defter `notebooks/01_Gereksinimler_ve_Veri_Kontrolu.ipynb` dosyasıdır. Açılan defterde `Knowledge Tracing (proje)` çekirdeğini seçin. Dosyalar proje klasöründe saklanır.

Yeni bir bilgisayarda PowerShell üzerinden `./setup_environment.ps1` ortamı hazırlar. NVIDIA ekran kartı bulunmayan bilgisayarda `./setup_environment.ps1 -CpuOnly` kullanılabilir. Kurulum ayrı `.venv` klasörüne yapılır.

## Belgeler

- [Ana proje raporu](output/pdf/Proje_Raporu.pdf)
- [Kod ve gerçek çıktı açıklamaları eki](output/pdf/Kod_ve_Cikti_Aciklamalari.pdf)
- [Teslimi inceleme kılavuzu](docs/teslim_kilavuzu.md)
- [Gerçek zamanlı oturum kullanımı](docs/gercek_zamanli_oturum.md)

- [Makale ile yöntem uyumu](docs/makale_yontem_uyumu.md)
- [Düzeltilmiş yöntemin deney sonuçları](docs/duzeltilmis_yontem_sonuclari.md)

- [Proje gereksinimleri ve kabul ölçütleri](docs/gereksinimler.md)
- [Bloklar ve veri akışı](docs/bloklar_ve_veri_akisi.md)
- [Uygulama ve deney planı](docs/uygulama_plani.md)
- [Yeniden veri denetimi ve öğrenci ayrımı](docs/veri_denetimi.md)
- [İlk sürümün ana deney sonuçları](docs/ana_deney_sonuclari.md)
- [Öğrenci takibi, davranış grupları ve örnek oturum](docs/ogrenci_takibi_ve_oruntuler.md)
- [Metin, ses ve görsel etkileşim örnekleri](docs/metin_ses_gorsel_ornekleri.md)
- [Kavram sözlüğü](GLOSSARY.md)

İlk defterin ardından `notebooks/02_Veri_Hazirlama_ve_Ogrenci_Ayrimi.ipynb` dosyasını açın. Bu defter öğrenci ayrımını, eğitimden hesaplanan soru zorluğunu ve sabit referans kümelerini gösterir.

Üçüncü defter `notebooks/03_LKM_Modeli_ve_Kucuk_Deneme.ipynb`, çalışan LKM ve üç uyarlama yöntemini, küçük eğitim denemesini ve güvenilirlik kontrollerini gösterir. [Model yöntemi](docs/model_yontemi.md) kapasite, maskeleme, eğitilen bileşenler ve makaleden uygulama farklarını açıklar. Küçük deneme ana test karşılaştırması değildir.

[Küçük deneme sonuçları](docs/pilot_sonucu.md), doğrulama ölçülerini ve donanım ölçümlerini içerir. İlk sürümde tam veri üzerinde üç başlangıçlı karşılaştırma tamamlandı. `notebooks/04_Ana_Deneyler_ve_Karsilastirma.ipynb` dosyası dokuz koşulun sonuçlarını, ham tahminlerin denetimini ve grafikleri gösterir. İlk sürümde promptsuz yöntem daha yüksek ortalama test AUC değeri elde etti; bütün sonuçlar ve uygulama sınırları ayrı sonuç belgesinde yer alır.

Beşinci defter `notebooks/05_Ogrenci_Gostergeleri_ve_Davranis_Gruplari.ipynb`, sabit referans sorularındaki beceri tahminini ve 2550 eğitim öğrencisinin 9598 geçmiş penceresinden öğrenilmiş 2 davranış grubunu gösterir. Gruplar öğrencinin yakın dönem durumunu açıklar; kalıcı öğrenme tarzı olarak sunulmaz.

Altıncı defter `notebooks/06_Ornek_Oturum_ve_Calisma_Onerileri.ipynb`, ilk beş cevap sonrasında 20 gerçek kaydı sırayla işler. Cevaptan önce tahmin, cevaptan sonra beceri göstergesi, grup ve gerekçeli mevcut soru önerisi gösterilir. `step_number` değeri değiştirilerek bir adım incelenebilir. Model ağırlıkları oturumda sabittir; gerçek kaynak takvim zamanı bulunmadığı için ilerleme cevap adımlarıyla gösterilir.

Örnek oturum için düzeltilmiş KMPT metin tarifi içinde yalnızca doğrulama BCE değeriyle başlangıç seçildi. Prototip aynı hedef zaman bloğu dikkat yorumunu kullanır; bu politika testten önce belirlenmiştir. Güncel seçim `outputs/main/paper_alignment_v2/interpretations/prototype_selection.json`, etkin model yönlendirmesi `configs/active_model.json` içindedir. Eski öğrenci gösterimleri `outputs/archive/pre_alignment_v1/` altında saklanır.

Yedinci defter `notebooks/07_Metin_Ses_ve_Gorsel_Etkilesimler.ipynb`, küçük çalışan metin, ses ve görsel örneklerini gösterir. Üç Türkçe çalışma isteği üç giriş türünde hazırlanmıştır. Whisper gerçekten konuşmayı yazıya çevirir, Windows OCR görseldeki yazıyı okur, küçük denetimli sınıflandırıcı mevcut beceriyi belirler ve aynı öğrenci geçmişiyle mevcut soru önerilir. İstekler doğru veya yanlış cevap olarak geçmişe eklenmez. Tanıma hataları ham çıktılarda korunur. Defterde bir metin veya dosya değiştirilerek yeni kısa girdi denenebilir. Açıklamalı rapor ve kod/çıktı eki output/pdf klasöründedir; teslimi inceleme kılavuzu docs/teslim_kilavuzu.md içindedir.

Bu bilgisayarda ses modeli indirildi ve Türkçe OCR çalışması doğrulandı. Yeni bilgisayarda proje ortamı kurulduktan sonra `.venv/Scripts/python.exe scripts/download_interaction_model.py` ses ağırlıklarını sabit sürümde bir kez indirir. Sonraki ses işlemleri yereldir, API anahtarı gerekmez. Görsel örneği Windows Türkçe OCR dilini kullanır. Hazır sentetik WAV dosyalarını çalıştırmak için sesleri yeniden üretmek gerekmez.

## Gerçek zamanlı oturum

Dokuzuncu defter `notebooks/09_Gercek_Zamanli_Oturum_Takibi.ipynb`, yeni cevapları gerçek alım zamanlarıyla kaydeden Jupyter ekranıdır. Yeni oturum boş başlar. Beceri ve bilinen soru kimliğini seçip **Soruyu başlat** düğmesine basın; ardından dışarıda değerlendirilmiş doğru veya yanlış sonucu kaydedin. Göstergeler, grup ve öneri güncellenir; yatay eksen gerçek kayıt saatidir. Oturum kimliğini saklayarak kaydı yeniden açabilirsiniz. Kaynakta soru metinleri bulunmadığından bu ekran yeni soru sunumu veya otomatik cevap değerlendirmesi yapmaz.

Yeni kayıtlar `outputs/live/sessions/` altında ayrı saklanır. Arşivdeki `outputs/live/demo_session.json` ve arayüz kontrol kaydı, açıkça **otomatik işlev testi** olarak etiketlenmiştir; gerçek öğrenci öğrenme verisi değildir. İki cevap arasında son tahmin tutulur; geçen süreyi yeni bir KT özelliği veya öğrenme kazanımı olarak kullanmayız. Model ve geçmiş deney dosyaları sabittir.

## Kapsam

Ana deneyler promptsuz uyarlama, KPT ve KMPT karşılaştırmasıdır. Öğrenme örüntüleri denetimsiz gruplamayla incelenir; çalışma önerileri tahminlerden ve gözlenen geçmişten üretilir. Örnek oturum mevcut kayıtların adım adım yeniden oynatımıdır. Metin, ses ve görsel için küçük çalışan örnekler, öğrencinin ifade ettiği çalışma isteğini mevcut beceriye bağlar. Bunlar çok modlu verilerle yeniden eğitilmiş bilgi takibi modeli veya genel konuşma tanıma başarısı olarak sunulmaz.

Kapsamlı yeni soru bankası ve yeni yazılmış sorular için model uyarlama kapsam dışındadır. Önceki proje defterleri, ağırlıkları ve sonuçları kullanılmaz. Ham ASSIST2009 verisi yeniden incelenir ve model kodu bu proje için yazılır.

## Çıktılar

`outputs/audit/` veri ve ortam kontrollerini, `outputs/figures/` diyagramları ve grafikleri içerir. Güncel ana deney ayarları, ağırlıkları, dönem geçmişleri ve örnek bazındaki tahminleri `outputs/main/paper_alignment_v2/` altındadır. Katı yorumun özeti `analysis/`, aynı hedef zaman bloğu kontrolü ve etkin prototip seçimi `interpretations/` bölümündedir. `outputs/main/three_seed_v1/` ilk uygulamanın tarihsel deney kaydı, `outputs/pilot/` ise küçük ilk denemedir.

Öğrenilmiş davranış grupları ve eğitim pencereleri `outputs/patterns/`, gerçek kayıtlardan üretilen örnek oturum `outputs/replay/session.json`, güvenilirlik kontrolleri ve test çıktıları `outputs/validation/` altındadır. Grafikler PNG ve düzenlenebilir SVG olarak saklanır. Bu kayıtlar önerilerin öğrenme kazanımı sağladığını ölçen bir müdahale deneyi değildir.

Küçük etkileşim girdileri ve yazılmış ifade derlemi `data/interactions/`, yerel ses modeli `data/models/whisper-base/`, gerçek tanıma çıktıları ve sınıflandırıcı ölçüleri `outputs/interactions/` altındadır. Model dosya özetleri ile kaynak sürümü ayrıca saklanır. `scripts/try_interaction.py` tek bir metin, WAV veya görsel girdisini defter dışında denemek için de kullanılabilir.

Yöntem düzeltmesi deneyi `outputs/main/paper_alignment_v2/` altında ayrı saklanır. Üç başlangıçta promptsuz, KPT ve KMPT ana koşulları ile tek şablon yerleşimi kontrolü olmak üzere 12 koşul vardır. Makaledeki maskeden sonra prompt sırası ile iki modülde katı nedensellik arasındaki gerilim duyarlılık deneyiyle ayrıca kaydedilir. Eski test ayrımı yeniden kullanıldığından yeni sonuçlar tamamen yeni ve dokunulmamış doğrulayıcı test gibi sunulmaz.
