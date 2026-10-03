"""Freeze new cases before inference; never select examples by model success."""
import hashlib
import json
import random
import re
from collections import Counter
from .common import ROOT, read_json, write_json, fingerprint, digest
from .alternatives_jev import read_rows
from .industry import question

# Authored as contrastive business examples, not SME-validated operational data.
# Every tuple is one independent scenario with paired EN/NB renderings.
BUSINESS = {
'routing': [
('it','The vibration trend page for P-417 shows HTTP 403 after my role changed. Please restore access; the pump itself is running normally.', 'Trendsiden for vibrasjon på P-417 viser HTTP 403 etter at rollen min ble endret. Gjenopprett tilgangen; selve pumpen går normalt.'),
('documents','No repair is requested. Please send the signed pressure-test certificate for the replacement spool.', 'Det er ikke bestilt reparasjon. Send det signerte trykkprøvingssertifikatet for den nye rørspolen.'),
('procurement','We have the approved drawing and the correct part number. Please obtain a commercial offer for six seal kits.', 'Vi har godkjent tegning og riktig delenummer. Innhent et kommersielt tilbud på seks tetningssett.'),
('maintenance','The historian and network are healthy. A technician needs to inspect the loose connector on the cabinet door.', 'Historikksystemet og nettverket virker. En tekniker må undersøke den løse kontakten på skapdøren.'),
('review','Please renew my document-system password and buy two pressure gauges for the workshop.', 'Forny passordet mitt i dokumentsystemet og kjøp to manometre til verkstedet.'),
('it','The inspection app exports an empty PDF even though all readings are saved. Please fix the export function.', 'Inspeksjonsappen eksporterer en tom PDF selv om alle målingene er lagret. Rett eksportfunksjonen.'),
('documents','The supplier is not being asked for a price. We need the latest approved wiring diagram for this serial number.', 'Vi ber ikke leverandøren om pris. Vi trenger siste godkjente koblingsskjema for dette serienummeret.'),
('procurement','The test certificate is accepted. Can you arrange a three-week rental of a calibrated torque wrench?', 'Testsertifikatet er akseptert. Kan dere ordne tre ukers leie av en kalibrert momentnøkkel?'),
('maintenance','Do not change the software settings. Please replace the damaged keyboard on the operator console.', 'Ikke endre programvareinnstillingene. Bytt det skadede tastaturet på operatørkonsollen.'),
('review','Something seems wrong with the compressor record. Please sort it out.', 'Noe virker galt med kompressorregistreringen. Ordne det, takk.'),
('it','The maintenance planner can open the work order but cannot save changes after yesterday\'s application update.', 'Vedlikeholdsplanleggeren kan åpne arbeidsordren, men får ikke lagret endringer etter gårsdagens programoppdatering.'),
('documents','We already own the spare. Find the approved assembly procedure before we schedule the replacement.', 'Vi eier allerede reservedelen. Finn godkjent monteringsprosedyre før vi planlegger utskiftingen.'),
('procurement','Ignore the old quotation attached for reference. Request a new fixed-price offer for machining this flange.', 'Se bort fra det gamle tilbudet som følger som referanse. Be om et nytt fastpristilbud på maskinering av denne flensen.'),
('maintenance','The login issue was resolved yesterday. The remaining request is to inspect the cooling fan that has stopped turning.', 'Innloggingsproblemet ble løst i går. Det gjenstående oppdraget er å undersøke kjøleviften som har sluttet å rotere.'),
('review','Please send the calibration history and have a technician investigate the unstable reading.', 'Send kalibreringshistorikken og få en tekniker til å undersøke den ustabile målingen.'),
('review','Can you book a meeting room and order lunch for the design review?', 'Kan dere bestille møterom og lunsj til designgjennomgangen?'),
],
'documents': [
('design','Required geometry: shaft diameter 35.00 mm, tolerance h6. The inspection report shall record the measured diameter after manufacture.', 'Påkrevd geometri: akseldiameter 35,00 mm, toleranse h6. Inspeksjonsrapporten skal angi målt diameter etter produksjon.'),
('inspection','Actual shaft diameter measured at three positions: 34.989, 34.992 and 34.990 mm. Instrument ID M-17. Inspector signed 12 September.', 'Faktisk akseldiameter målt i tre posisjoner: 34,989, 34,992 og 34,990 mm. Instrument-ID M-17. Kontrollør signerte 12. september.'),
('purchase','Order line 20: four actuator kits at NOK 7,200 each. Supplier shall include an inspection certificate. Payment due 30 days after delivery.', 'Ordrelinje 20: fire aktuatorsett til NOK 7 200 per stykk. Leverandøren skal inkludere inspeksjonssertifikat. Betaling 30 dager etter levering.'),
('work','Job 6421: replace the actuator kit, record labour hours and return the removed unit to the workshop. Reference purchase order PO-882.', 'Jobb 6421: bytt aktuatorsettet, registrer arbeidstimer og returner den demonterte enheten til verkstedet. Referanse innkjøpsordre PO-882.'),
('design','Material specification: wetted parts shall be 316L. The supplier quotation is referenced only to identify the product family.', 'Materialspesifikasjon: medieberørte deler skal være 316L. Leverandørtilbudet er bare referert for å identifisere produktfamilien.'),
('inspection','Calibration performed against reference R-9. As-found error +0.8%; as-left error +0.1%. Work order WO-110 is referenced for traceability.', 'Kalibrering utført mot referanse R-9. Avvik før justering +0,8 %; etter justering +0,1 %. Arbeidsordre WO-110 er referert for sporbarhet.'),
('purchase','We offer machining of 12 flanges for a total of EUR 3,600. Offer valid until 30 October. Drawing D-19 defines the dimensions.', 'Vi tilbyr maskinering av 12 flenser for totalt EUR 3 600. Tilbudet gjelder til 30. oktober. Tegning D-19 angir dimensjonene.'),
('work','Completed job entry: cleaned filter housing, replaced element and restored service. Two technician hours booked. Inspection certificate attached.', 'Registrering av utført jobb: renset filterhus, byttet element og satt tilbake i drift. To teknikertimer ført. Inspeksjonssertifikat vedlagt.'),
('design','Bill of materials for assembly A-16: one housing, two bearings and one retaining ring. No purchase quantities or prices are authorized by this list.', 'Stykkliste for sammenstilling A-16: ett hus, to lagre og én låsering. Listen godkjenner ingen innkjøpsmengder eller priser.'),
('inspection','Visual examination results: no surface cracks found on the three sampled welds. Photo references and inspector identity follow.', 'Resultat av visuell kontroll: ingen overflatesprekker funnet på de tre undersøkte sveisene. Fotoreferanser og kontrollørens identitet følger.'),
('purchase','Requisition: procure 20 replacement filters, charged to cost centre 440. Maintenance job WO-619 explains the demand.', 'Rekvisisjon: anskaff 20 erstatningsfiltre, belast kostnadssted 440. Vedlikeholdsjobb WO-619 forklarer behovet.'),
('work','Task card: inspect the belt, adjust tension if needed and enter completion status. Use drawing D-81 for the guard arrangement.', 'Oppgavekort: undersøk remmen, juster strammingen ved behov og registrer ferdigstatus. Bruk tegning D-81 for utforming av vernet.'),
('design','Design basis: this package uses two independent power feeds. This document specifies the required arrangement; test results are issued separately.', 'Designgrunnlag: pakken bruker to uavhengige strømtilførsler. Dokumentet spesifiserer påkrevd løsning; testresultater utgis separat.'),
('inspection','Factory test results: channel A responded in 42 ms and channel B in 45 ms. Values were recorded against specification S-63 revision C.', 'Resultat fra fabrikktest: kanal A reagerte på 42 ms og kanal B på 45 ms. Verdiene ble registrert mot spesifikasjon S-63 revisjon C.'),
('purchase','Service agreement: supplier provides monthly calibration visits at an agreed hourly rate. Travel reimbursement and invoicing terms are listed below.', 'Serviceavtale: leverandøren utfører månedlige kalibreringsbesøk til avtalt timepris. Reiserefusjon og fakturavilkår er oppført nedenfor.'),
('work','Planned maintenance package: reserve a technician for four hours, remove the cover and inspect the coupling. Return this card with completion notes.', 'Planlagt vedlikeholdspakke: reserver en tekniker i fire timer, demonter dekselet og undersøk koblingen. Returner kortet med ferdigmelding.'),
]}

def normalize(s):
    return re.sub(r'\s+', ' ', s).strip().casefold()

def leaves(v):
    if isinstance(v, str): yield v
    elif isinstance(v, dict):
        for x in v.values(): yield from leaves(x)
    elif isinstance(v, list):
        for x in v: yield from leaves(x)

def build():
    import laya
    from .claims_data import download
    tables, provenance = download()
    old = read_json(ROOT/'data/prepared/claims.json')
    excluded = set()
    prior_files = {}
    for path in (ROOT/'data/prepared').glob('*.jsonl'):
        # Intern's not-yet-run full reproduction contains the entire AG News
        # test set. It is a planned claim fixture, not prior model evidence.
        if path.stem in ['intern_claims','jev_live'] or path.stem.startswith('jev_fresh') or 'preflight' in path.stem: continue
        prior_files[path.name] = digest(path)
        for r in read_rows(path):
            for text in leaves(r.get('state', r.get('text', ''))):
                if len(text.strip()) >= 20: excluded.add(normalize(text))
    for suite in old['suites'].values():
        for r in suite['cases']:
            excluded.update(normalize(t) for t in leaves(r['state']) if len(t.strip()) >= 20)
    prior_files['claims.json'] = digest(ROOT/'data/prepared/claims.json')
    rows = []
    def add(suite, sid, state, qs, gold, **meta):
        row = dict(id=suite+'/'+sid, suite=suite, state=state, questions=qs, gold=gold, **meta)
        row['input_sha256'] = fingerprint({'state': state, 'questions': qs})
        rows.append(row)
    for key in ['news','emotion','spam','phishing']:
        template = old['suites'][key+'.replication']['cases'][0]['question']
        labels = old['suites'][key+'.replication']['labels']
        candidates = []; used = set()
        oldids = {int(c['id']) for name,s in old['suites'].items() if name.startswith(key+'.') for c in s['cases']}
        for i,r in enumerate(tables[key]):
            if i in oldids: continue
            if key == 'news': state={'article':r['text']}; gold=labels[r['label']]
            elif key == 'emotion': state={'text':r['text']}; gold=labels[r['label']]
            elif key == 'spam': state=laya.email_state(r.get('subject') or '', (r.get('message') or '')[:3000]); gold=labels[r['label']]
            else:
                if not (r.get('Email Text') or '').strip() or r.get('Email Type') not in ['Safe Email','Phishing Email']: continue
                state={'email':r['Email Text'][:3000]}; gold=str(r['Email Type']=='Phishing Email').lower()
            strings=[normalize(t) for t in leaves(state) if len(t.strip())>=20]
            if not strings or any(t in excluded or t in used for t in strings): continue
            used.update(strings)
            candidates.append((i,state,gold,max(strings,key=len)))
        # Natural remaining-class prevalence; deterministic sample before predictions.
        random.Random(20261001).shuffle(candidates)
        assert len(candidates)>=300,(key,len(candidates))
        for j,(i,state,gold,text) in enumerate(candidates[:300]):
            split='development' if j<100 else 'test'
            add('fresh/'+key, str(i), state, {'decision':template}, {'decision':[gold]},
                split=split, family=hashlib.sha256(text.encode()).hexdigest(), source_index=i,
                source=provenance[key], selection='New to this evaluation, not guaranteed unseen during model training')
    for task, cases in BUSINESS.items():
        for i,(label,en,nb) in enumerate(cases):
            split='development' if i<4 else 'test'
            family=f'fresh_business/{task}/{i}'
            for language,text in [('en',en),('nb',nb)]:
                qs=question(task, language, 'contextual')
                # Make the policy identical across languages in meaning, explicit on all models.
                if task=='routing' and language=='en':
                    qs['decision']['instructions']='Route by the action currently requested, not equipment merely mentioned or already resolved issues. Choose review for unrelated, unclear or multiple requests. Which team should handle it?'
                elif task=='routing':
                    qs['decision']['instructions']='Velg etter handlingen som etterspørres nå, ikke utstyr som bare nevnes eller problemer som allerede er løst. Velg review for irrelevante, uklare eller flere ulike forespørsler. Hvilken gruppe skal behandle saken?'
                add(f'fresh_business/{task}/{language}',str(i),text,qs,{'decision':[label]},
                    split=split,family=family,language=language,sector='engineering operations',
                    label_provenance='Assistant-authored bilingual scenario and label; SME validation pending')
    dest=ROOT/'data/prepared/jev_fresh.jsonl'
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if dest.exists(): assert dest.read_text(encoding='utf-8')==payload,'Frozen fixture changed'
    else: dest.write_text(payload,encoding='utf-8')
    counts=Counter((r['suite'],r['split']) for r in rows)
    protocol={'fixture_sha256':digest(dest),'n':len(rows),'heads':len(rows),'suites':dict(Counter(r['suite'] for r in rows)),
              'split_counts':{'/'.join(k):v for k,v in counts.items()},'frozen_before_model_calls':True,
              'seed':20261001,'source_revisions':provenance,'exclusion_file_sha256':prior_files,
              'selection':'Entire news, emotion, spam and phishing task groups selected after earlier local results; 300 new cases each, excluding previous source indices and normalized prior input text. 100 development + 200 test, sampled without model outputs. The separately planned, not-yet-run full Intern claim fixture includes the entire AG News test corpus and is excluded from prior-evidence exclusions; its overlapping results are not additional independent confirmation. Public training exposure unknown. Phishing dataset labels may include broad spam; results do not establish credential-theft detection.',
              'business':'32 new bilingual scenario families (64 records), eight development families and 24 test families; assistant-authored labels, not real operating records or SME validation. Each language reported separately; translations are paired, not independent observations.',
              'metrics':'Per-task test accuracy including failures; macro F1, confusion matrices, paired family bootstrap 95% intervals; Brier/NLL/ECE on valid predictions with coverage. No pooled headline ranking. Decision thresholds selected on development only, then applied unchanged to test.',
              'business_scope':'Queue suggestions, document filing suggestions and screening for human review. No equipment control, work-permit approval or autonomous safety decision.'}
    write_json(ROOT/'results/alternatives/jev_fresh-protocol.json',protocol)
    print('Frozen',len(rows),'new cases:',dict(counts),flush=True)

if __name__=='__main__': build()
