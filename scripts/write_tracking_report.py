"""Gerçek gruplama ve oturum kayıtlarından yöntem ve sonuç açıklaması üret."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    meta = json.loads((ROOT / "outputs/patterns/metadata.json").read_text(encoding="utf-8"))
    session = json.loads((ROOT / "outputs/replay/session.json").read_text(encoding="utf-8"))
    integrity = json.loads((ROOT / "outputs/validation/student_tracking_integrity.json").read_text(encoding="utf-8"))
    states = [session["initial"]] + [event["after"] for event in session["events"]]
    selection = session["model_selection"]
    if selection.get("format") == "plkt_method_v2":
        model_config = json.loads((ROOT / "outputs/main/paper_alignment_v2/config.json").read_text())["model"]
    else:
        model_config = json.loads((ROOT / "outputs/main/three_seed_v1/config.json").read_text())["model"]
    changes = sum(left["behavior"]["cluster"] != right["behavior"]["cluster"] for left, right in zip(states, states[1:]))
    lines = ["# Öğrenci Takibi Davranış Grupları ve Örnek Oturum", "",
             "Sistem, eğitilmiş KMPT modeliyle beceri göstergeleri üretir; eğitim geçmişlerinden öğrendiği gruplara güncel öğrenci durumunu atar ve mevcut sorular arasından gerekçeli çalışma önerisi seçer. Örnek oturum gerçek kayıtları sırayla yeniden oynatır. Model ağırlıkları ve referans sorular oturum boyunca sabittir.", "",
             f"Etkin dikkat politikası {selection.get('inference_policy', 'legacy_causal_lkm_bidirectional_prompt')}. Güncel aynı hedef zaman bloğu yorumu yazarların doğrulanmış maskesi değildir; katı token referansı ve ayrı tanısal maske değerlendirmeleri sekizinci defterde korunur.", "",
             "## Beceri göstergeleri", "",
             f"Her becerinin başarı göstergesi, önceden eğitim sorularından seçilmiş sabit referans kümesindeki doğru cevap olasılıklarının ortalamasının yüz katıdır. Küme üyeleri ve eşit ağırlıkları oturumda değişmez. Tahmin yalnızca o ana kadar gözlenmiş en fazla {model_config['max_history']} etkileşimi kullanır.", "",
             "Gözlenen doğru cevap oranı ayrı hesaplanır. Grafiklerde son 20 etkileşim içinde ilgili beceriyi içeren cevaplar sayılır; soru sayısı da gösterilir. Tek bir doğru cevap yüzde 100 gözlenen oran üretebilir. Bu, becerinin kesin öğrenildiği anlamına gelmez.", "",
             "## Davranış gruplama girdileri", "",
             "Yakın dönem davranış vektöründe dört özellik vardır: son 20 cevaptaki doğru oranı, yakın geçmişte en sık görülen en fazla üç becerinin sabit referans tahmin ortalaması, pozitif cevap sürelerinin medyanının logaritmik dönüşümü ve cevap başına ortalama ipucu kullanımı. Aktif beceri ortalaması gruplama için bağlamsal bir özettir; genel öğrenme yüzdesi olarak sunulmaz.", "",
             f"Süre sınırları yalnızca eğitimden belirlenmiştir: alt sınır {meta['limits']['duration_lower_seconds']:.3f}, üst sınır {meta['limits']['duration_upper_seconds']:.3f} saniye. Sıfır ve geçersiz süreler eksik kabul edilir. İpucu sayıları eğitimden belirlenen {meta['limits']['hint_upper']:.1f} üst sınırında kesilir. Eksikler eğitim medyanlarıyla tamamlanır; ölçekleme eğitimde öğrenilir.", "",
             "## Grupların öğrenilmesi", "",
             f"{meta['training_students']} eğitim öğrencisinden, öğrenci başına en fazla dört ve en az beş cevap içeren toplam {meta['training_snapshots']} geçmiş penceresi kullanıldı. Doğrulama ve test öğrencileri grup, ölçek veya sınır öğrenimine alınmadı.", "",
             "KMeans için k means plus plus başlangıcı, 10 farklı merkez başlangıcı ve sabit 2026 rastgele başlangıcı kullanıldı. İki ile altı grup denendi. Her grubun pencerelerin en az yüzde 5'ini içerdiği adaylar arasından, eğitimden örneklenen en fazla 2000 penceredeki siluet değeri en yüksek olan seçildi.", "",
             "| Grup sayısı | Siluet | En küçük grup oranı yüzde | Büyüklük koşulu |",
             "|---|---|---|---|"]
    for row in meta["candidates"]:
        lines.append(f"| {row['clusters']} | {row['silhouette']:.4f} | {100*row['min_cluster_fraction']:.2f} | {'Uygun' if row['size_constraint_passed'] else 'Uygun değil'} |")
    lines += ["", f"Seçilen grup sayısı {meta['selected_clusters']}; siluet değeri {meta['selected_silhouette']:.4f}. Siluet, grupların bu özelliklerdeki ayrışmasını özetler; sınıflandırma doğruluğu veya öğrenme kazanımı yüzdesi değildir.", "",
              "| Grup | Pencere | En az bir penceresi bu grupta olan öğrenci | Gözlenen doğru oranı yüzde | Referans tahmini yüzde | Tipik cevap süresi saniye | Cevap başına ipucu |",
              "|---|---|---|---|---|---|---|"]
    for row in meta["profiles"]:
        lines.append(f"| {row['cluster']} | {row['snapshots']} | {row['students_with_snapshot_in_group']} | "
                     f"{100*row['mean_observed_accuracy']:.2f} | {100*row['mean_reference_probability']:.2f} | "
                     f"{row['median_response_seconds']:.2f} | {row['mean_hints']:.2f} |")
    lines += ["", "Aynı öğrencinin farklı dönemlerde farklı gruplarda pencereleri bulunabilir; öğrenci sütunu gruplar arasında toplanamaz. Grup numaraları başarı sırası veya kalıcı öğrenme tarzı değildir. Tahmin özelliği eğitim geçmişlerinde aynı modelin göstergelerinden üretildiği için bu gruplama ayrıca bağımsız bir genelleme başarısı ölçümü olarak yorumlanmaz.", "",
              "## Çalışma önerileri", "",
              "Öneri, mevcut eğitim sorularından bilinen soru kimlikleri arasından seçilir. Yakın geçmişteki aktif beceriler arasında sabit referans başarısı en düşük olan önceliklendirilir. Daha önce cevaplanmamış aday varsa tekrar sorusu seçilmez. En fazla 256 aday, zorluk aralığını kapsayacak şekilde alınır.", "",
              "Normal durumda tahmini doğru cevap olasılığı yüzde 65'e yakın bir soru seçilir. Son üç cevap yanlışsa hedef yüzde 80'e çıkarılarak daha erişilebilir; son üç cevap doğruysa yüzde 55'e indirilerek daha zorlayıcı bir çalışma hedeflenir. Bunlar önceden belirlenmiş açık kurallardır; öğrenilmiş öneri politikası veya pedagojik olarak kanıtlanmış en iyi eşikler değildir.", "",
              "Çıktı mevcut bir soru kimliği, beceri, zorluk, tahmini olasılık ve gerekçedir. Kapsamlı soru bankası hazırlanmaz, yeni yazılmış sorulara model uyarlaması yapılmaz. Davranış grubu öğreticiye ek bağlam sağlar; öğrenciye değişmez bir kişisel özellik atamaz.", "",
              "## Gerçek kayıtlarla örnek oturum", "",
              f"Örnek, yeterli uzunlukta doğrulama geçmişleri arasından yalnızca kayıt uzunluğu ve sabit rastgele başlangıçla seçildi. İlk {session['initial']['history_count']} cevap başlangıç geçmişidir; sonraki {len(session['events'])} gerçek cevap sırayla işlendi. Öğrenci, sonucun iyi görünmesine veya eğrinin artmasına göre seçilmedi.", "",
              "Her adımda hedef cevap okunmadan tahmin üretildi. Ardından kayıttaki cevap geçmişe eklendi; referans göstergeleri, gözlenen oranlar, grup ve öneri güncellendi. Yenilenen şey öğrenci geçmişidir; model ağırlıkları çevrimiçi yeniden eğitilmez.", "",
              f"Bu örnekte gruplar arasında {changes} geçiş görüldü. Göstergeler artabilir, azalabilir veya sabit kalabilir. Çizgiler ayrık cevap adımlarını birleştirir; aralarda gerçekleşmiş sürekli öğrenme ölçümü değildir.", "",
              "### Oturum sonundaki göstergeler", "",
              "| Beceri | Sabit referans tahmini yüzde | Son pencerede gözlenen yüzde | Bu orana dayanak cevap sayısı | Referans soru sayısı |",
              "|---|---|---|---|---|"]
    for row in states[-1]["skill_scores"]:
        observed = f"{row['observed_correct_percent']:.2f}" if row["observed_correct_percent"] is not None else "Yok"
        lines.append(f"| {row['name']} | {row['predicted_success_percent']:.2f} | {observed} | {row['recent_skill_answers']} | {row['reference_question_count']} |")
    lines += ["", "Kaynakta tekil adı güvenle eşlenemeyen ek kavramlar beceri kodlarıyla gösterilir. Yakın dönem oranının yükselmesi, eski yanlış cevapların pencere dışına çıkmasından da kaynaklanabilir. Tahmini gösterge ile gözlenen oran aynı ölçü değildir.", "",
              "### Öneri gerekçesi örnekleri", "",
              f"Başlangıç: {session['initial']['recommendation']['reason']}", "",
              f"Son adım: {states[-1]['recommendation']['reason']}", "",
              "Yeniden oynatımda bir sonraki gerçek kayıt, önerilen sorudan farklı olabilir. Bu akış önerinin uygulandığı veya öğrenmeyi artırdığı bir müdahale deneyi değildir.", "",
              "## Zaman bilgisi", "",
              f"Gösterim başlangıcı {session['display_started_at']}; son adımın gösterim saati {session['events'][-1]['display_timestamp']}. Bunlar bu program çalıştırmasının gerçek saatleridir. Kaynakta geçmiş olayların takvim tarihleri bulunmadığı için kaynak zamanları boş bırakılmıştır. Grafiklerin birincil ekseni etkileşim adımıdır; geçmiş öğrencinin günlere yayılmış çalışma veya unutma süresi üretilmez.", "",
              "## Güvenilirlik denetimi", "",
              f"{integrity['snapshots_checked']} eğitim penceresinin özellikleri yeniden hesaplandı; grup seçimi, ölçeklerin eğitim istatistikleri, gerçek cevap eşleşmeleri ve öneri kuralları doğrulandı. Bilgi takibi ağırlıkları, grup modeli ve referans kümesi oturumda değişmedi. Önceki ana deneyin model kaynakları ve sonuçları korunuyor.", "",
              "Kayıtlar outputs/patterns, outputs/replay ve outputs/validation/student_tracking_integrity.json altında bulunur. Güncel yöntem seçimi oturum kaydının model_selection alanındadır. Düzeltilmiş yöntem için [yeni deney sonuçları](duzeltilmis_yontem_sonuclari.md), önceki sürüm için [ilk deney sonuçları](ana_deney_sonuclari.md) ayrı saklanır.", "",
              "## Yöntem kaynakları", "",
              "Gruplama ve siluet uygulaması için [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [siluet ölçüsü](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) ve [eksik değer tamamlama](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) belgeleri esas alınmıştır."]
    (ROOT / "docs/ogrenci_takibi_ve_oruntuler.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Student tracking method and results report generated.")


if __name__ == "__main__":
    main()
