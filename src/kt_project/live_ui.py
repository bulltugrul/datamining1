"""Jupyter'da yeni oturum, değerlendirilmiş cevap girişi ve gerçek saat grafiği."""

from datetime import datetime, timezone
import html
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .live_session import LiveSession, parse_time


def timeline_figure(record, until=None):
    """Olaylar arasında son tahmini tut; ara öğrenme değeri uydurma."""
    states = [record['initial_state']]+[event['after'] for event in record['events']]
    times = [parse_time(record['initial_state_recorded_at'])]+[
        parse_time(event['state_updated_at']) for event in record['events']]
    end = parse_time(record['closed_at']) if record['closed_at'] else (until or datetime.now(timezone.utc))
    if end.tzinfo is None or end < times[-1]:
        raise ValueError('Grafik bitiş zamanı son olaydan önce olamaz.')
    colors = ['#2463a6','#c56b14','#22846b','#845da3','#536878','#b14e6f']
    fig, axes = plt.subplots(2,1,figsize=(9,6.3),sharex=True,layout='constrained')
    for index, skill in enumerate(record['tracked_skills']):
        values = [next((row['predicted_success_percent'] for row in state['skill_scores']
            if row['skill']==skill),float('nan')) for state in states]
        name = next((row['name'] for row in states[-1]['skill_scores'] if row['skill']==skill),str(skill))
        axes[0].step(times+[end],values+[values[-1]],where='post',label=name,color=colors[index%len(colors)])
        axes[0].scatter(times,values,s=16,color=colors[index%len(colors)])
    observed = [state['observed_recent_correct_ratio'] for state in states]
    values = [100*value if value is not None else float('nan') for value in observed]
    axes[1].step(times+[end],values+[values[-1]],where='post',color='#52647c',label='Gözlenen son 20 cevap oranı')
    axes[1].scatter(times,values,s=19,color='#52647c')
    for ax in axes:
        ax.set(ylim=(-4,104),ylabel='Yüzde')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='best')
    axes[0].set_title('Sabit referans sorularında tahmini başarı')
    axes[1].set_title('Cevap sayısı: '+str(len(record['events']))+' (başlangıç geçmişi yok)')
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y\n%H:%M:%S',tz=ZoneInfo('Europe/Istanbul')))
    axes[1].set_xlabel('Gerçek kayıt zamanı (Europe/Istanbul); cevaplar arasında son değer tutulur')
    mode = 'Otomatik işlev testi; insan öğrenme verisi değil' if record['response_origin']=='automated_smoke_test' else 'Elle girilmiş değerlendirilmiş cevaplar'
    fig.suptitle(mode,fontsize=11)
    return fig


def event_table(record):
    return pd.DataFrame([{'Cevap':event['sequence'],'Mevcut soru':event['original_problem_id'],
        'Tahmin %':100*event['predicted_correct_probability'],'Doğru (1/0)':event['correct'],
        'Tahmin saati':parse_time(event['predicted_at']).astimezone(ZoneInfo('Europe/Istanbul')).isoformat(timespec='milliseconds'),
        'Cevap alım saati':parse_time(event['answer_received_at']).astimezone(ZoneInfo('Europe/Istanbul')).isoformat(timespec='milliseconds'),
        'Cevap süresi sn':event['response_duration_seconds'],'Kaynak':event['response_origin']}
        for event in record['events']])


def create_live_controls(resources, *, origin='manual_assessed_response', directory=None):
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    label = widgets.Text(value='Katılımcı',description='Etiket:')
    saved_id = widgets.Text(description='Oturum ID:',placeholder='Kaydedilmiş oturum kimliği',layout=widgets.Layout(width='95%'))
    start = widgets.Button(description='Yeni oturum',button_style='primary')
    resume = widgets.Button(description='Kaydı aç')
    skill = widgets.Dropdown(options=[(resources.names.get(s,str(s)),s) for s in resources.default_skills],
        description='Beceri:',layout=widgets.Layout(width='95%'))
    question = widgets.Dropdown(description='Soru:',layout=widgets.Layout(width='95%'))
    prepare = widgets.Button(description='Soruyu başlat')
    correct = widgets.Button(description='Doğru cevabı kaydet',button_style='success')
    wrong = widgets.Button(description='Yanlış cevabı kaydet',button_style='warning')
    hints = widgets.BoundedIntText(value=0,min=0,max=100,description='İpucu:')
    refresh = widgets.Button(description='Görünümü yenile')
    close = widgets.Button(description='Oturumu kapat')
    message = widgets.HTML()
    output = widgets.Output()
    current = {'session':None}

    def questions(*_):
        choices = [entry for entry in resources.catalog.values() if skill.value in entry['skills']]
        choices.sort(key=lambda entry:entry['question'])
        question.options = [(str(resources.inverse_questions[entry['question']]),resources.inverse_questions[entry['question']])
                            for entry in choices]
        question.value = question.options[0][1] if question.options else None
    skill.observe(questions,names='value')
    questions()

    def sync_enabled():
        session=current['session']
        record=session.snapshot() if session else None
        pending=record['pending'] if record else None
        opened=bool(record and record['status']=='open')
        prepare.disabled=not opened or pending is not None
        correct.disabled=wrong.disabled=not opened or pending is None
        close.disabled=not opened or pending is not None
        start.disabled=resume.disabled=opened
        saved_id.disabled=opened
        skill.disabled=question.disabled=pending is not None
        refresh.disabled=False

    def render(note=''):
        session = current['session']
        record = session.snapshot() if session else None
        pending = record['pending'] if record else None
        text = html.escape(note)
        if record:
            saved_id.value = record['session_id']
            text += '<br>Oturum: '+html.escape(record['session_id'])+' | Durum: '+{'open':'açık','closed':'kapalı'}[record['status']]
            if pending:
                text += f"<br>Açık soru {pending['original_problem_id']}: cevap öncesi tahmin %{100*pending['predicted_correct_probability']:.1f}. Sonuç henüz geçmişe eklenmedi."
        message.value = '<p>'+text+'</p>'
        with output:
            clear_output(wait=True)
            if record:
                state = record['events'][-1]['after'] if record['events'] else record['initial_state']
                display(pd.DataFrame(state['skill_scores'])[['name','predicted_success_percent',
                    'observed_correct_percent','recent_skill_answers','reference_question_count']].rename(columns={
                    'name':'Beceri','predicted_success_percent':'Tahmini başarı %',
                    'observed_correct_percent':'Gözlenen doğru %','recent_skill_answers':'Cevap sayısı',
                    'reference_question_count':'Referans soru sayısı'}))
                if state['behavior']:
                    print('Yakın dönem davranışı:',state['behavior']['description'])
                else:
                    print('Davranış grubu için en az',state['behavior_minimum_answers'],'cevap gerekir.')
                recommendation=state['recommendation']
                if recommendation['available']:
                    print('Önerilen mevcut soru:',recommendation['original_problem_id'])
                    name=resources.names.get(recommendation['skill'],'Seçilen beceri')
                    print(recommendation['reason'].replace(f"Beceri {recommendation['skill']}",name,1))
                fig=timeline_figure(record)
                display(fig)
                plt.close(fig)
                if record['events']:
                    display(event_table(record).tail(10).round(3))
        # Grafik ve tablolar yayımlanmadan yeni giriş açılmaz.
        sync_enabled()

    def guard(action):
        def handler(_):
            for control in (start,resume,prepare,correct,wrong,close,refresh):
                control.disabled=True
            try:
                action()
                render('Kayıt güncellendi.')
            except Exception as error:
                message.value='<p style="color:#a02929">'+html.escape(str(error))+'</p>'
            finally:
                sync_enabled()
        return handler

    def begin():
        current['session']=LiveSession(resources,label.value,origin=origin,directory=directory)
    def reopen():
        current['session']=LiveSession.resume(resources,saved_id.value.strip(),directory=directory)
    def answer(value):
        session=current['session']
        session.answer(session.snapshot()['pending']['question_token'],value,hints.value)
        hints.value=0
    start.on_click(guard(begin))
    resume.on_click(guard(reopen))
    prepare.on_click(guard(lambda:current['session'].prepare(question.value)))
    correct.on_click(guard(lambda:answer(1)))
    wrong.on_click(guard(lambda:answer(0)))
    close.on_click(guard(lambda:current['session'].close()))
    refresh.on_click(guard(lambda:None))
    render('Yeni oturum başlatın veya kaydedilmiş oturumu açın.')
    banner = ('<h3>Gerçek zamanlı cevap kaydı</h3><p>Mevcut soruyu öğrenciye dışarıda sunup değerlendirilmiş sonucunu girin. '
        'Kaynak veri soru metni içermediği için bu ekran soruları kimlikleriyle gösterir. '
        'Yeni oturum boş başlar; eski bir öğrencinin cevapları aktarılmaz. '
        'Tahmin, kesin öğrenme yüzdesi değildir.</p>')
    if origin=='automated_smoke_test':
        banner += '<p><b>Otomatik arayüz testi: kayıtlar insan cevabı değildir.</b></p>'
    panel=widgets.VBox([widgets.HTML(banner),label,widgets.HBox([start,resume]),saved_id,skill,question,prepare,
        hints,widgets.HBox([correct,wrong]),widgets.HBox([refresh,close]),message,output])
    # Testler gerçek düğme geri çağrılarını kullanır; kullanıcı ekranına teknik nesneler eklenmez.
    panel._live_controls={'start':start,'prepare':prepare,'correct':correct,'wrong':wrong,'close':close,
        'refresh':refresh,'resume':resume,'saved_id':saved_id,'question':question,'message':message,'current':current}
    return panel
