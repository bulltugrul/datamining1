# Akıllı Öğretim ve Bilgi Takibi

Bu sözlük, prompt tuning ile bilgi takibi projesindeki kavramların ortak anlamlarını tanımlar. Öğrenciye ait başarı ölçüleri ile tahmin modelinin değerlendirme ölçülerini birbirinden ayırır.

## Language

**Bilgi takibi (Knowledge tracing)**:
Öğrencinin geçmiş öğrenme etkileşimlerinden hareketle zaman içinde değişen bilgi durumunun modellenmesi ve gelecekteki sorulara vereceği cevapların tahmin edilmesidir.

**Öğrenme etkileşimi**:
Bir öğrencinin belirli bir soruya verdiği cevaba ilişkin, öğrenme geçmişinde bir sıraya veya zamana sahip olaydır.

**Bilgi kavramı (Knowledge concept)**:
Bir sorunun ölçtüğü konu veya beceridir. Bir soru birden fazla bilgi kavramıyla ilişkili olabilir.

**Ders**:
Birden fazla bilgi kavramını kapsayan öğretim alanıdır. Matematik bir dersken kesirler ve denklemler bu dersin bilgi kavramlarıdır.
_Avoid_: Tek bir beceriyi belirtmek için ders

**Soru zorluğu**:
Bir sorunun doğru cevaplanmasının güçlüğünü ifade eden özelliktir. Öğrencinin o soruyu doğru cevaplama olasılığıyla aynı kavram değildir.

**Etkileşim sırası**:
Bir öğrencinin öğrenme etkileşimlerinin birbirinden önce veya sonra gerçekleşme düzenidir. Etkileşimler arasında geçen gerçek süreyi tek başına belirtmez.

**Olay zamanı**:
Bir öğrenme etkileşiminin gerçekleştiği tarih ve saattir. Bir cevap için harcanan süre veya etkileşimin sıra numarasıyla aynı kavram değildir.

**Öğrenme oturumu**:
Bir öğrencinin belirli bir çalışma süreci içinde gerçekleştirdiği soru ve cevap etkileşimlerinin bütünüdür.

**Örnek öğrenme oturumu**:
Öğrenci takibinin işleyişini, cevaplar geldikçe değişen tahmin ve göstergelerle görünür kılan örnek öğrenme akışıdır. Bu akış tek başına gerçek öğrencilerin öğrenme kazanımlarına ilişkin kanıt oluşturmaz.

**Kayıt yeniden oynatımı**:
Önceden kaydedilmiş öğrenci etkileşimlerinin sıraları korunarak, her etkileşimin ardından oluşan öğrenci durumunun yeniden izlendiği akıştır.

**Gösterim zamanı**:
Bir etkileşimin örnek oturumda işlenip gösterildiği tarih ve saattir. Geçmiş bir kaydın kaynakta bulunmayan gerçek olay zamanının yerine geçmez.

**Doğru cevap olasılığı**:
Öğrencinin geçmişi ve hedef soru için modelin ürettiği doğru cevap verme olasılığı tahminidir. Öğrencinin konunun belirli bir yüzdesini kesin olarak öğrendiğinin ölçümü değildir.
_Avoid_: Kesin öğrenme yüzdesi

**Gözlenen doğru cevap oranı**:
Tanımlanmış bir soru, konu veya zaman aralığında öğrencinin doğru cevaplarının değerlendirilen cevaplarına oranıdır.
_Avoid_: Model doğruluğu

**Referans soru kümesi**:
Bir becerideki tahmini başarıyı zaman içinde karşılaştırmak için içeriği sabit tutulan soru kümesidir. Öğrencinin bir sonraki çalışacağı sorularla aynı küme olmak zorunda değildir.

**Beceri bazında tahmini başarı yüzdesi**:
Öğrencinin güncel öğrenme geçmişi için, ilgili becerinin sabit referans soru kümesindeki doğru cevap olasılıklarının ortalamasının yüzdeyle ifadesidir. Bir konunun kesin olarak öğrenilmiş bölümünü göstermez.
_Avoid_: Kesin öğrenme yüzdesi, modelin tahmin doğruluğu

**Etkileşim türü**:
Öğrenci ile öğretici arasındaki bilginin metin, ses veya görsel biçiminde iletilmesidir. Farklı etkileşim türleri aynı beceri veya soruyla ilişkili olabilir.

**Öğrenme örüntüsü**:
Öğrencinin yakın dönem cevap geçmişinde başarı, cevaplama süresi ve mevcutsa ipucu kullanımı bakımından görülen davranış düzenidir. Öğrencinin değişmez bir öğrenme tarzını veya kişisel özelliğini ifade etmez.

**Öğrenci davranış grubu**:
Benzer yakın dönem öğrenme örüntülerini paylaşan öğrenci geçmişlerinin oluşturduğu gruptur. Bir öğrenci farklı dönemlerde farklı gruplarla ilişkilendirilebilir.

**Çalışma önerisi**:
Öğrencinin güncel durumuna göre sunulan, çalışılacak beceri, sonraki soru veya açıklamalı örnek yönlendirmesidir. Tek başına önerinin öğrenme kazanımı sağladığına ilişkin kanıt değildir.

**Öneri gerekçesi**:
Bir çalışma önerisinin hangi öğrenci geçmişine, başarı tahminine veya gözlenen davranışa dayandığını açıklayan bilgidir.

**Modelin tahmin doğruluğu**:
Değerlendirme örneklerinde modelin doğru veya yanlış cevap tahminlerinin gerçekleşen cevap sonuçlarıyla uyuşma oranıdır. Tek bir öğrencinin bilgi düzeyini ifade etmez.
_Avoid_: Öğrencinin öğrenme yüzdesi

**Soft prompt**:
Bir modeli hedef göreve yönlendirmek amacıyla öğrenilen sürekli sayısal temsillerden oluşan prompttur. İnsan tarafından yazılmış sabit bir talimat metniyle aynı kavram değildir.

**Prompt tuning**:
Modelin hedef göreve uyarlanmasında öğrenilebilir prompt temsillerinin eğitildiği yaklaşımdır.

**KPT (Knowledge-guided Prompt Template)**:
Hedef sorunun bilgi kavramlarını kullanarak öğrencinin cevabını tahmin etmeye yönelik prompt türüdür.

**KMPT (Knowledge-guided Multi-Prompt Templates)**:
Cevap tahmini ile yardımcı soru tahmini görevlerini farklı prompt türleriyle bir araya getiren yaklaşımdır.

**LKM (Large Knowledge Model)**:
Öğrencilerin geçmiş öğrenme etkileşimlerindeki kavramları ve ilişkileri ön eğitimle temsil eden bilgi modelidir.

**RTM (Random Token Masking)**:
Öğrenme geçmişinde gizlenen soru ve cevap bilgilerini geri bulmayı amaçlayan özdenetimli görevdir.

**FBO (Forward-Backward Order)**:
İki etkileşim parçasının özgün öğrenci geçmişinde doğru ardışıklıkla bir araya gelip gelmediğini belirleyen özdenetimli görevdir.

**PromEnc (Prompt Encoder)**:
Öğrenilebilir promptları ve hedefe ait soru, kavram ve cevap bileşenlerini ortak bir temsil alanına dönüştüren prompt kodlayıcısıdır.
