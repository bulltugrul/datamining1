"""İlk kontrol defterini ve blok diyagramını yeniden üret."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def draw_diagram() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 10), dpi=160)
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    blue, orange, gray = "#deebff", "#ffead0", "#eef1f5"

    def box(x, y, w, h, title, subtitle, color, output=False):
        if output:
            patch = Polygon([(x + .18, y), (x + w, y), (x + w - .18, y + h),
                             (x, y + h)], closed=True, facecolor=color,
                            edgecolor="#52647c", linewidth=1.3)
        else:
            patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                   facecolor=color, edgecolor="#52647c", linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*.66, title, ha="center", va="center",
                fontsize=11.5, weight="bold", color="#142a43", linespacing=1.3)
        ax.text(x + w/2, y + h*.24, subtitle, ha="center", va="center",
                fontsize=9.5, color="#324960", linespacing=1.3)

    def arrow(points, label=None, label_at=None):
        if len(points) > 2:
            ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                    color="#52647c", linewidth=1.5)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=13, color="#52647c", linewidth=1.5))
        if label:
            ax.text(*label_at, label, fontsize=9, color="#324960", ha="center",
                    bbox={"facecolor": "#f7f9fc", "edgecolor": "none", "pad": 2})

    ax.text(.5, 9.5, "Akıllı öğretim sisteminin öğrenme blokları", fontsize=23,
            color="#142a43", weight="bold")
    ax.text(.5, 9.03, "Mavi: denetimli görevler   |   Turuncu: denetimsiz görev   |   Gri: veriler ve hesaplanan çıktılar",
            fontsize=11, color="#52647c")

    box(.6, 7.15, 3.6, 1.1, "Sesi metne çevirme", "Hazır denetimli model", blue)
    box(.6, 5.6, 3.6, 1.1, "Görselden yazı çıkarma", "Hazır denetimli model", blue)
    box(5.4, 6.2, 4.2, 1.25, "İfade edilen beceriyi\nsınıflandırma", "Denetimli öğrenme", blue)
    ax.text(7.5, 8.25, "Yazılı ifade", fontsize=12, ha="center", color="#324960")
    arrow([(7.5, 8.0), (7.5, 7.45)])
    arrow([(4.2, 7.7), (4.8, 7.7), (4.8, 7.0), (5.4, 7.0)])
    arrow([(4.2, 6.15), (4.8, 6.15), (4.8, 6.6), (5.4, 6.6)])
    ax.text(12.5, 8.0, "Küçük metin, ses ve görsel örnekleri", fontsize=12,
            ha="center", color="#324960")
    ax.text(12.5, 7.57, "Yeni soru bankası veya yeni sorulara uyarlama yok", fontsize=9.5,
            ha="center", color="#52647c")

    box(.6, 3.5, 3.6, 1.2, "ASSIST2009", "Geçmiş etkileşimler ve hedef soru", gray, True)
    box(5.4, 3.5, 4.2, 1.2, "Sonraki cevabı tahmin", "Denetimli öğrenme\nPromptsuz uyarlama · KPT · KMPT", blue)
    box(10.8, 3.5, 4.3, 1.2, "Başarı göstergeleri", "Olasılık ve sabit referans ortalaması", gray, True)
    arrow([(4.2, 4.1), (5.4, 4.1)])
    arrow([(9.6, 4.1), (10.8, 4.1)])

    box(.6, 1.0, 3.6, 1.25, "Yakın dönem özellikleri", "Başarı · süre · mevcutsa ipucu", gray, True)
    box(5.4, 1.0, 4.2, 1.25, "Davranışları gruplama", "Denetimsiz öğrenme", orange)
    box(10.8, 1.0, 4.3, 1.25, "Grafik ve çalışma önerileri", "Göstergeler, grup ve açık seçim kuralları", gray, True)
    arrow([(2.4, 3.5), (2.4, 2.25)])
    arrow([(12.0, 3.5), (12.0, 2.85), (3.4, 2.85), (3.4, 2.25)])
    arrow([(4.2, 1.62), (5.4, 1.62)])
    arrow([(9.6, 1.62), (10.8, 1.62)])
    arrow([(14.0, 3.5), (14.0, 2.25)])
    arrow([(9.6, 6.82), (15.6, 6.82), (15.6, 1.62), (15.1, 1.62)],
          "İfade edilen beceri", (12.2, 6.95))
    ax.text(.6, .38, "Ön eğitim: yalnızca eğitim verisiyle RTM ve FBO özdenetimli görevleri.  Öneri seçimi ve grafik hesapları ayrı öğrenme modeli değildir.",
            fontsize=9.5, color="#52647c")
    fig.tight_layout(pad=.6)
    output = ROOT / "outputs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "ogrenme_bloklari.png", dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(output / "ogrenme_bloklari.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_notebook() -> None:
    md = nbf.v4.new_markdown_cell
    code = lambda text: nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Knowledge Tracing (proje)",
                              "language": "python", "name": "kt-project"}
    nb.metadata.language_info = {"name": "python", "version": "3.13"}
    nb.cells = [
        md("# Gereksinimler ve Veri Kontrolü\n\nBu defter proje kapsamını, çalışma ortamını, öğrenme bloklarını ve ham ASSIST2009 dosyasının yeniden denetimini gösterir. Model eğitimi ve bilimsel karşılaştırmalar sonraki aşamadadır. Önceki proje kodu veya ağırlıkları kullanılmaz."),
        md("## Proje klasörünü ve araçları hazırlama\n\nDefter kök klasörden veya notebooks klasöründen çalıştırılabilir. Projeye ait yardımcı kodlar src klasöründedir."),
        code('''
            from pathlib import Path
            import json
            import sys
            from IPython.display import display, Markdown, Image
            import pandas as pd

            ROOT = Path.cwd().resolve()
            if not (ROOT / "src" / "kt_project").exists():
                ROOT = ROOT.parent
            assert (ROOT / "src" / "kt_project").exists(), "Proje klasörü bulunamadı."
            if str(ROOT / "src") not in sys.path:
                sys.path.insert(0, str(ROOT / "src"))
            print("Proje:", ROOT.name)
        '''),
        md("## Python ve ekran kartı kontrolü\n\nEkran kartının görünmesine ek olarak CUDA üzerinde bir ileri hesap ve geri yayılım yapılır. Bu küçük işlem bir bilgi takibi modeli eğitimi değildir."),
        code('''
            from kt_project.environment import check_environment

            environment = check_environment()
            display(pd.DataFrame(environment["packages"].items(), columns=["Paket", "Sürüm"]))
            display(pd.DataFrame([{
                "Python": environment["python_version"],
                "CUDA": environment["cuda_build"],
                "Ekran kartı": environment["gpu_name"],
                "Bellek GB": environment["gpu_memory_gib"],
                "CUDA hesaplama kontrolü": environment["cuda_calculation_passed"],
            }]))
        '''),
        md("## Proje gereksinimleri\n\nAşağıdaki tablo görev duyuruları ve birlikte kabul edilen kapsamı ayırır. Göstergeler, örnek oturum ve deneylerin kabul ölçütleri de belgede bulunur."),
        code('''
            display(Markdown((ROOT / "docs" / "gereksinimler.md").read_text(encoding="utf-8")))
        '''),
        md("## Öğrenme blokları\n\nÖğrenme görevleri kutularla, girdi ve çıktılar farklı şekillerle gösterilir. Küçük etkileşim örneklerinin metinleri, tarihsel öğrenci verisi olarak değerlendirilmez."),
        code('''
            display(Image(filename=str(ROOT / "outputs" / "figures" / "ogrenme_bloklari.png")))
        '''),
        md("## Ham veri kaynağını doğrulama\n\nDüzeltilmiş Skill Builder dosyası data/raw altında ham haliyle saklanır. İçerik özeti dosyanın sonradan değişip değişmediğini denetler. Yalnızca toplu sayılar gösterilir."),
        code('''
            from kt_project.data_audit import audit_assist2009, file_digest

            raw_path = ROOT / "data" / "raw" / "skill_builder_data_corrected_collapsed.csv"
            manifest = json.loads((raw_path.parent / "manifest.json").read_text(encoding="utf-8"))
            assert file_digest(raw_path) == manifest["sha256"], "Ham dosyanın içeriği değişmiş."
            print("Dosya:", raw_path.name)
            print("SHA256:", manifest["sha256"])
            print("Kaynak:", manifest["source_page"])
            audit = audit_assist2009(raw_path)
        '''),
        md("## Kayıt ve beceri kapsamı\n\nUygunluk ön incelemesi; geçerli cevap, kimlik ve beceri bilgisi olan olayları, çelişkili kopyaları ve en az beş etkileşim koşulunu dikkate alır. Eğitim ve test ayrımı bu defterde henüz yapılmaz."),
        code('''
            raw = audit["raw"]
            eligible = audit["eligibility_preview"]
            display(pd.DataFrame({
                "Ölçü": ["Kayıt veya olay", "Öğrenci", "Soru", "Beceri"],
                "Ham dosya": [raw["raw_rows"], raw["students"], raw["questions"], raw["skills"]],
                "Uygunluk sonrası": [eligible["retained_events"], eligible["retained_students"],
                                      eligible["retained_questions"], eligible["retained_skills"]],
            }))
            display(pd.DataFrame([
                {"Kontrol": "Eksik veya geçersiz beceri satırı", "Sayı": raw["missing_or_invalid_skill_rows"]},
                {"Kontrol": "Çok becerili satır", "Sayı": raw["multi_skill_rows"]},
                {"Kontrol": "Geçersiz cevap etiketi", "Sayı": raw.get("invalid_binary_label_rows", 0)},
                {"Kontrol": "Çelişkili olay", "Sayı": raw["conflicting_events"]},
            ]))
        '''),
        md("## Zaman ve zorluk alanlarının anlamı\n\norder_id bir sıralama alanıdır. ms_first_response cevap süresidir; takvim zamanı değildir. Hazır soru zorluğu alanı bulunmadığı için zorluk, öğrenci ayrımı yapıldıktan sonra yalnızca eğitim cevaplarından hesaplanacaktır."),
        code('''
            display(pd.DataFrame([
                {"Alan": "order_id / position", "Anlam": "Etkileşim sırası"},
                {"Alan": "ms_first_response / overlap_time", "Anlam": "Süre bilgisi; olay tarihi değil"},
                {"Alan": "Örnek oturumun gösterim zamanı", "Anlam": "Yeniden oynatım saati; geçmiş olay zamanı değil"},
                {"Alan": "Soru zorluğu", "Anlam": "Sonraki aşamada yalnızca eğitim verisinden hesaplanacak"},
            ]))
            assert eligible["retained_events"] > 0
            assert eligible["min_events_per_student"] >= 5
            assert eligible["retained_students"] <= raw["students"]
            assert not audit["fields"]["documented_calendar_timestamp_available"]
            print("Ham veri uygunluk kontrolleri tamamlandı.")
        '''),
        md("## Sonraki aşama\n\nÖğrenci bazında eğitim, doğrulama ve test ayrımı; çok becerili olayların temsili; eğitimden soru zorluğu hesaplama; LKM ön eğitimi ve üçlü model karşılaştırması. Bu defter model doğruluğu veya öğrenme kazanımı sonucu üretmez."),
    ]
    output = ROOT / "notebooks" / "01_Gereksinimler_ve_Veri_Kontrolu.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.validate(nb)
    nbf.write(nb, output)
    print(output.name)


def make_data_notebook() -> None:
    md = nbf.v4.new_markdown_cell
    code = lambda text: nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Knowledge Tracing (proje)",
                              "language": "python", "name": "kt-project"}
    nb.cells = [
        md("# Öğrenci Ayrımı ve Eğitim Verisinden Özellikler\n\nBu defter veri hazırlama aşamasını yeniden çalıştırır. Öğrenciler ayrılır; kimlik sözlükleri, soru zorluğu ve sabit referans soruları yalnızca eğitim bölümünden belirlenir. Model eğitimi bu defterin kapsamı değildir."),
        code('''
            from pathlib import Path
            import sys
            import json
            import numpy as np
            import pandas as pd
            from IPython.display import display

            ROOT = Path.cwd().resolve()
            if not (ROOT / "src" / "kt_project").exists():
                ROOT = ROOT.parent
            assert (ROOT / "src" / "kt_project").exists()
            if str(ROOT / "src") not in sys.path:
                sys.path.insert(0, str(ROOT / "src"))
        '''),
        md("## Veri hazırlamayı yeniden çalıştırma\n\nÇok becerili bir gerçek cevap tek etkileşim olarak tutulur. Öğrenciler yüzde 70 eğitim, yüzde 15 doğrulama ve kalan yüzde 15 test olacak şekilde ayrılır. Ayrım başlangıcı 2026 olarak sabittir."),
        code('''
            from kt_project.prepare_data import prepare_dataset

            metadata = prepare_dataset(
                ROOT / "data" / "raw" / "skill_builder_data_corrected_collapsed.csv",
                ROOT / "data" / "processed", split_seed=2026)
            table = pd.DataFrame(metadata["splits"]).T
            table.index.name = "Grup"
            display(table.rename(columns={
                "students": "Öğrenci", "interactions": "Etkileşim",
                "unseen_question_interactions": "Eğitimde görülmeyen soru içeren etkileşim",
                "interactions_with_unseen_skill": "Eğitimde görülmeyen beceri içeren etkileşim"}))
        '''),
        md("## Öğrenci kümelerinin ayrılığını kontrol etme\n\nAynı öğrenciye ait kayıtlar farklı gruplara dağılmaz. Test öğrencilerinin erken cevapları, o öğrencinin sonraki tahminleri için gözlenen geçmiş olabilir; model ağırlıklarını eğitmek için kullanılamaz."),
        code('''
            prepared = ROOT / "data" / "processed"
            splits = json.loads((prepared / "student_splits.json").read_text(encoding="utf-8"))
            for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                assert not set(splits[left]).intersection(splits[right])
                print(left, "ve", right, "öğrenci kesişimi: 0")
        '''),
        md("## Zorluğun yalnızca eğitimden hesaplandığını kontrol etme\n\nBilinen sorunun zorluğu eğitimdeki hata oranıdır. Eğitimde bulunmayan soru için genel eğitim hata oranı kullanılır. Öğrenci bazında ayırma nedeniyle ortaya çıkan bu eksik kimlik durumu, yeni soru bankası geliştirmekten farklıdır ve raporda ayrıca belirtilir."),
        code('''
            with np.load(prepared / "interactions.npz", allow_pickle=False) as packed:
                arrays = {name: packed[name] for name in packed.files}
            training = arrays["split"] == 0
            training_error = float(1-arrays["correct"][training].mean())
            assert np.isclose(training_error, metadata["global_training_difficulty"])
            assert not (arrays["question"][training] == 1).any()
            unknown = arrays["question"] == 1
            assert np.allclose(arrays["difficulty"][unknown], training_error)
            display(pd.DataFrame([{
                "Eğitimden hesaplanan genel hata oranı": training_error,
                "Tüm etkileşimler": len(arrays["correct"]),
                "Bir olayda en fazla beceri": metadata["max_skills_per_event"],
            }]))
        '''),
        md("## Sabit referans sorularının kapsamı\n\nHer beceri için eğitim sorularından, farklı zorlukları kapsayan en fazla 32 referans soru seçilir. Göstergenin değişimi öğrenci geçmişinden kaynaklanır; referans kümesi oturumda değiştirilmez."),
        code('''
            references = json.loads((prepared / "references.json").read_text(encoding="utf-8"))
            counts = pd.Series({skill: len(items) for skill, items in references.items()}, dtype="int64")
            display(counts.describe().to_frame("Beceri başına referans soru sayısı"))
            train_questions = set(arrays["question"][training].tolist())
            assert all(item["question"] in train_questions
                       for items in references.values() for item in items)
            print("Referans beceri sayısı:", len(references))
            print("Bütün referans sorular eğitim sorularından seçilmiş.")
        '''),
        md("## Eğitimde görülmeyen sorular\n\nBu etkileşimler testten sessizce çıkarılmaz. Soru için UNK kimliği, olayda verilen bilinen beceriler ve eğitimden hesaplanan genel zorluk kullanılır. Sonuçlarda tüm örnekler ve bilinen soru alt grubu ayrı gösterilecektir."),
        code('''
            coverage = pd.DataFrame(metadata["splits"]).T
            coverage["Eğitimde görülmeyen soru oranı yüzde"] = (
                100 * coverage["unseen_question_interactions"] / coverage["interactions"])
            display(coverage[["Eğitimde görülmeyen soru oranı yüzde"]].round(3))
            print("Hazırlanan dosyalar: interactions.npz, metadata.json, references.json, vocabulary.json, student_splits.json")
        '''),
        md("## Sonraki aşama\n\nHazırlanan dizilerle küçük model denemesi yapılacak; hedef cevabın gizlenmesi, geleceğe erişimin engellenmesi ve prompt gradyanları kontrol edilecek. Ardından LKM ön eğitimi ve üç yöntem için bağımsız başlangıçlarla ana deneyler yürütülecek."),
    ]
    output = ROOT / "notebooks" / "02_Veri_Hazirlama_ve_Ogrenci_Ayrimi.ipynb"
    nbf.validate(nb)
    nbf.write(nb, output)
    print(output.name)


if __name__ == "__main__":
    draw_diagram()
    make_notebook()
    make_data_notebook()
