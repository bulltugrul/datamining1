"""Yöntem düzeltmesi sonrası güncel ve tarihsel defterleri açıkça ayır."""

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTICE = "Bu belge önceki uygulama sürümünü açıklar. Yöntem denetimi sonrasındaki güncel uygulama için [makale uyumu](makale_yontem_uyumu.md) ve [düzeltilmiş sonuçlar](duzeltilmis_yontem_sonuclari.md) esas alınmalıdır. Önceki sayılar tarihsel kayıt olarak korunmuştur."


def main():
    meta = json.loads((ROOT / "outputs/patterns/metadata.json").read_text(encoding="utf-8"))
    for filename in ("model_yontemi.md", "ana_deney_sonuclari.md", "pilot_sonucu.md"):
        path = ROOT / "docs" / filename
        content = path.read_text(encoding="utf-8")
        if NOTICE not in content:
            first, rest = content.split("\n", 1)
            path.write_text(first+"\n\n"+NOTICE+"\n"+rest, encoding="utf-8")
    for filename in ("03_LKM_Modeli_ve_Kucuk_Deneme.ipynb", "04_Ana_Deneyler_ve_Karsilastirma.ipynb"):
        path = ROOT / "notebooks" / filename
        notebook = nbformat.read(path, as_version=4)
        marker = "**Tarihsel sürüm kaydı.**"
        if marker not in notebook.cells[0].source:
            notebook.cells[0].source += ("\n\n"+marker+" Bu defter ilk uygulamanın çıktısını korur. "
                "Yöntem denetimi ve düzeltilmiş ana sonuçlar 08_Makale_Yontem_Denetimi_ve_Duzeltilmis_Deneyler.ipynb içindedir. "
                "Öğrenci takibi için 05–07 defterleri güncel ağırlıklarla yeniden çalıştırılmıştır.")
            nbformat.write(notebook, path)
    path = ROOT / "README.md"
    content = path.read_text(encoding="utf-8")
    introduction = ("Güncel yöntem, makaleyle eşitlik ve dikkat düzeyinde denetlenmiş ikinci sürümdür. "
        "Ana sonuçlar `notebooks/08_Makale_Yontem_Denetimi_ve_Duzeltilmis_Deneyler.ipynb` içindedir. "
        "Öğrenci takibi ve etkileşimler için 05–07 defterleri yeni ağırlıklarla yeniden çalıştırılır; "
        "03–04 defterleri ilk sürümün tarihsel kaydıdır. Bu çalışma ölçekli yöntem uygulamasıdır; "
        "makaledeki bütün deneylerin birebir tekrarı olarak sunulmaz.")
    if introduction not in content:
        content = content.replace("## Çalışma ortamı", introduction+"\n\n## Çalışma ortamı", 1)
    content = content.replace("- [Üç başlangıçlı ana deney sonuçları](docs/ana_deney_sonuclari.md)",
        "- [İlk sürümün ana deney sonuçları](docs/ana_deney_sonuclari.md)")
    if "[Makale ile yöntem uyumu]" not in content:
        content = content.replace("## Belgeler\n", "## Belgeler\n\n- [Makale ile yöntem uyumu](docs/makale_yontem_uyumu.md)\n- [Düzeltilmiş yöntemin deney sonuçları](docs/duzeltilmis_yontem_sonuclari.md)\n", 1)
    content = content.replace("Tam veri üzerinde üç başlangıçlı ana karşılaştırma da tamamlandı.",
        "İlk sürümde tam veri üzerinde üç başlangıçlı karşılaştırma tamamlandı.")
    content = content.replace("Bu kurulumda promptsuz yöntem daha yüksek", "İlk sürümde promptsuz yöntem daha yüksek")
    content = content.replace("9598 geçmiş penceresinden öğrenilmiş iki davranış grubunu", 
        f"9598 geçmiş penceresinden öğrenilmiş {meta['selected_clusters']} davranış grubunu")
    old = "Örnek oturum için KMPT başlangıcı, önceden belirlenen yöntem içinde yalnızca doğrulama BCE değeriyle seçildi; seçim kaydı ana deney klasörünün `analysis/prototype_selection.json` dosyasındadır."
    new = ("Örnek oturum için düzeltilmiş KMPT metin tarifi içinde yalnızca doğrulama BCE değeriyle başlangıç seçildi. "
        "Prototip aynı hedef zaman bloğu dikkat yorumunu kullanır; bu politika testten önce belirlenmiştir. Güncel seçim `outputs/main/paper_alignment_v2/interpretations/prototype_selection.json`, etkin model yönlendirmesi "
        "`configs/active_model.json` içindedir. Eski öğrenci gösterimleri `outputs/archive/pre_alignment_v1/` altında saklanır.")
    content = content.replace(old, new)
    revision_note = ("Yöntem düzeltmesi deneyi `outputs/main/paper_alignment_v2/` altında ayrı saklanır. "
        "Üç başlangıçta promptsuz, KPT ve KMPT ana koşulları ile tek şablon yerleşimi kontrolü olmak üzere 12 koşul vardır. "
        "Makaledeki maskeden sonra prompt sırası ile iki modülde katı nedensellik arasındaki gerilim "
        "duyarlılık deneyiyle ayrıca kaydedilir. Eski test ayrımı yeniden kullanıldığından yeni sonuçlar "
        "tamamen yeni ve dokunulmamış doğrulayıcı test gibi sunulmaz.")
    if revision_note not in content:
        content += "\n"+revision_note+"\n"
    path.write_text(content, encoding="utf-8")
    print("Current and historical method documentation marked.")


if __name__ == "__main__":
    main()
