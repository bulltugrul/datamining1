"""Kalıcı gerçek zamanlı oturum ve Jupyter kullanım ekranı defteri."""

from pathlib import Path
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells=[md('''
    # Gerçek zamanlı oturum takibi

    Bu defter yeni cevap olaylarını gerçek alım saatleriyle kaydeder. Oturum boş
    başlar; ASSIST2009 öğrencisinin geçmişi yeni katılımcıya aktarılmaz. Mevcut
    sorunun doğruluğu dışarıda değerlendirilip doğru veya yanlış olarak girilir.
    Kaynak veri soru metni içermediğinden seçim mevcut soru kimliğiyle yapılır.
    Yeni soru bankası hazırlanmaz ve model yeniden eğitilmez.

    Aşağıdaki kayıtlar gerçek saat çağrılarıyla çalışan **otomatik işlev testidir**;
    insan öğrenme verisi veya öğrenme kazanımı deneyi değildir. En alttaki ekran
    elle girilecek değerlendirilmiş cevapları kaydetmek için kullanılabilir.
    '''),code('''
    from pathlib import Path
    import inspect
    import json
    import os
    import subprocess
    import sys
    import pandas as pd
    from IPython.display import display, Image, Code

    ROOT=Path.cwd().resolve()
    if not (ROOT / "src/kt_project").exists():
        ROOT=ROOT.parent
    sys.path.insert(0,str(ROOT/"src"))
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".runtime/matplotlib"))
    os.environ.setdefault("PYTHONUTF8","1")
    import torch
    torch.set_num_threads(4)
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if not (ROOT/"outputs/live/demo_session.json").exists():
        subprocess.run([sys.executable,str(ROOT/"scripts/run_live_session_demo.py")],cwd=ROOT,check=True)
    record=read(ROOT/"outputs/live/demo_session.json")
    assert record["response_origin"]=="automated_smoke_test"
    print("Örnek kaydın türü:",record["response_origin"])
    print("Başlangıç geçmişi:",record["initial_history_count"])
    print("Başlangıç zamanı UTC:",record["created_at"])
    '''),md('''
    ## Cevaptan önce tahmin ve kalıcı kayıt

    Soruyu başlatma çağrısı yalnızca soru kimliğini alır; hedef cevabı almaz.
    Bekleyen soru geçmişe eklenmez. Sonuç girilince zaman damgası ve süre
    kaydedilir; ardından geçmiş ve göstergeler güncellenir. Aynı cevap kimliği
    iki kez gönderilirse geçmişe ikinci kayıt eklenmez. Oturum diske atomik
    olarak yazılır; kimliğiyle yeniden açılabilir. İçerik özeti ve olay zinciri
    kazara değişmiş kayıtları tespit eder; kriptografik kimlik doğrulama değildir.
    '''),code('''
    from kt_project.live_session import LiveResources,LiveSession,parse_time
    from kt_project.live_ui import event_table,create_live_controls
    display(Code(inspect.getsource(LiveSession.prepare),language="python"))
    display(Code(inspect.getsource(LiveSession.answer),language="python"))
    display(event_table(record).round(3))
    assert all(event["history_count_before"]==index for index,event in enumerate(record["events"]))
    assert all(parse_time(event["predicted_at"])<=parse_time(event["answer_received_at"])
               <=parse_time(event["state_updated_at"]) for event in record["events"])
    '''),md('''
    ## Gerçek zaman ekseni ve son değeri tutma

    Yatay eksen gerçek kayıt zamanıdır. Her cevaptan sonra yeni model tahmini
    hesaplanır. İki olay arasında son tahmin tutulur; araya yapay öğrenme değeri
    eklenmez. İlk cevap öncesinde gözlenen doğru oranı yoktur. Davranış grubu
    en az beş yeni cevaptan sonra gösterilir. Geçen süre, KT modelinin yeni bir
    girdisi değildir; burada zamanla unutma modeli eğitilmemiştir.
    '''),code('''
    display(Image(filename=str(ROOT/"outputs/live/live_timeline.png")))
    display(pd.DataFrame(record["initial_state"]["skill_scores"]))
    display(pd.DataFrame(record["events"][-1]["after"]["skill_scores"]))
    display(pd.Series(read(ROOT/"outputs/validation/live_session_integrity.json")))
    '''),md('''
    ## Elle kullanılabilen oturum ekranı

    1. Katılımcı etiketiyle yeni oturum açın. Oturum kimliğini saklayın.
    2. Beceri ve mevcut soru kimliğini seçip **Soruyu başlat** düğmesine basın.
    3. Sorunun değerlendirilmiş sonucunu doğru veya yanlış olarak kaydedin.
       Bu düğmeler cevap doğruluğunu otomatik ölçmez.
    4. Göstergeyi ve gerçek saat grafiğini inceleyin; diğer soruyla devam edin.
    5. İş bitince oturumu kapatın. Sonradan oturum kimliğiyle kaydı açabilirsiniz.

    Canlı kayıtlar outputs/live/sessions klasöründe ayrı tutulur; eğitim,
    doğrulama ve test dosyaları değişmez. Aynı oturumu tek açık çalışma
    görünümünde yönetin. Eski bir görünüm güncel kaydın üstüne yazamaz.
    '''),code('''
    resources=LiveResources.load(ROOT)
    # Normal kullanımda elle değerlendirilmiş cevap kaydı seçilir. Arayüz test
    # süreci kendi ortamında otomatik kaynak etiketini açıkça verir.
    origin=os.environ.get("KT_LIVE_RESPONSE_ORIGIN","manual_assessed_response")
    live_panel=create_live_controls(resources,origin=origin)
    display(live_panel)
    '''),md('''
    ## İşlev testleri ve sınırlar

    Sabit test saatleri kullanan birim testleri kayıt mekanizmasını doğrular;
    gösterilen örnek oturumun saatleri ise gerçek duvar saatinden alınmıştır.
    Otomatik test cevapları insan cevabı olarak etiketlenmez. Gerçek katılımcı
    verisiyle uzun süreli öğrenme veya öneri etkisi çalışması yapılmamıştır.
    '''),code('''
    completed=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests",
        "-p","test_live_session.py","-v"],cwd=ROOT,capture_output=True,text=True,
        encoding="utf-8",errors="replace")
    print(completed.stdout+completed.stderr)
    completed.check_returncode()
    if (ROOT/"outputs/live/live_controls.png").exists():
        print("Aşağıdaki görüntü, otomatik kaynak etiketiyle gerçek Jupyter arayüzü kontrolünden alınmıştır.")
        display(Image(filename=str(ROOT/"outputs/live/live_controls.png")))
    ''')]
notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Knowledge Tracing (proje)','language':'python','name':'kt-project'},
    'language_info':{'name':'python','version':'3.13.2'}})
nbf.validate(notebook)
path=ROOT/'notebooks/09_Gercek_Zamanli_Oturum_Takibi.ipynb'
nbf.write(notebook,path)
print(path.name)
