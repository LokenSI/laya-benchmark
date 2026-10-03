"""Assistant-authored feasibility cases, not customer data or SME-validated gold.

Every line is a distinct scenario family with an English/Bokmal translation pair.
The first four routing / three document families per class are development only.
All remaining families are frozen holdout. No prompt is selected using holdout.
"""
from .common import ROOT, fingerprint, write_json

# label | sector | English | Norwegian Bokmal
ROUTING = """
maintenance|oil_gas|Pump P-101 is leaking at the seal. Please send a technician to investigate.|Pumpe P-101 lekker ved tetningen. Send en tekniker for å undersøke.
maintenance|manufacturing|The packaging conveyor stops every few minutes. We need someone to repair it.|Pakkebåndet stopper med få minutters mellomrom. Vi trenger noen som kan reparere det.
maintenance|engineering|The test rig motor will not start. Can maintenance check the equipment?|Motoren på testriggen starter ikke. Kan vedlikehold sjekke utstyret?
maintenance|manufacturing|A bearing on the press is making a grinding noise. Please arrange a physical inspection.|Et lager på pressen lager skrapelyder. Bestill en fysisk inspeksjon.
maintenance|oil_gas|Compressor C-204 vibrates more than yesterday. Could the rotating equipment team look at it?|Kompressor C-204 vibrerer mer enn i går. Kan teamet for roterende utstyr se på den?
maintenance|manufacturing|The pallet lift remains at floor level when the operator presses raise. It needs attention on site.|Palleløfteren blir stående på gulvet når operatøren trykker opp. Den må undersøkes på stedet.
maintenance|engineering|Water appears beneath the laboratory chiller after each run. Please find the source.|Det kommer vann under kjøleren i laboratoriet etter hver kjøring. Finn kilden.
maintenance|oil_gas|This is not a request for a new valve. The installed actuator does not move; please examine it.|Dette er ikke en bestilling av en ny ventil. Den monterte aktuatoren beveger seg ikke. Undersøk den.
maintenance|manufacturing|The cutter produces rough edges although the drawing is correct. Send someone to check the machine.|Kutteren lager grove kanter selv om tegningen er riktig. Send noen for å sjekke maskinen.
maintenance|engineering|The torque wrench no longer clicks at its set point. Can the workshop inspect it?|Momentnøkkelen klikker ikke lenger ved innstilt moment. Kan verkstedet undersøke den?
maintenance|oil_gas|The pressure transmitter reading is frozen even on the local display. Please inspect the instrument.|Trykktransmitteren viser en fast verdi også på det lokale displayet. Undersøk instrumentet.
maintenance|manufacturing|The replacement belt is already here. We need a fitter to install it on line 3.|Erstatningsremmen er allerede her. Vi trenger en mekaniker til å montere den på linje 3.
procurement|oil_gas|Please order two replacement seal kits for pump P-101.|Bestill to nye tetningssett til pumpe P-101.
procurement|manufacturing|We need a quotation for ten conveyor rollers for the warehouse.|Vi trenger et tilbud på ti transportbåndruller til lageret.
procurement|engineering|Purchase another calibrated torque wrench for the test bench.|Kjøp en ekstra kalibrert momentnøkkel til testbenken.
procurement|oil_gas|Ask our valve supplier for price and delivery time on four actuators.|Be ventilleverandøren om pris og leveringstid på fire aktuatorer.
procurement|manufacturing|Our last carton of filter elements has been used. Please replenish the stock with twenty units.|Den siste esken med filterelementer er brukt opp. Fyll opp lageret med tjue enheter.
procurement|engineering|We have the approved drawing. Find a vendor to manufacture six brackets and arrange the purchase.|Vi har godkjent tegning. Finn en leverandør som kan produsere seks braketter og ordne innkjøpet.
procurement|oil_gas|The compressor is operating normally. I only need the cost of a spare coupling for next year's budget.|Kompressoren fungerer normalt. Jeg trenger bare prisen på en reservekobling til neste års budsjett.
procurement|manufacturing|Could purchasing obtain three more of the stainless trays used on line 2?|Kan innkjøpsavdelingen skaffe tre flere av de rustfrie brettene som brukes på linje 2?
procurement|engineering|Please extend our CAD licence subscription for another year and request the supplier's offer.|Forny abonnementet på CAD-lisensene i ett år og be om tilbud fra leverandøren.
procurement|oil_gas|No repair visit is needed. Raise a requisition for the spare pressure sensor listed in the catalogue.|Det trengs ikke et reparasjonsbesøk. Opprett en rekvisisjon for reservetrykksensoren i katalogen.
procurement|manufacturing|The workshop has identified the required part. Please place the purchase order with the approved supplier.|Verkstedet har identifisert delen vi trenger. Send innkjøpsordren til godkjent leverandør.
procurement|engineering|Can you arrange the hire of a laser scanner for our survey next month?|Kan dere ordne leie av en laserskanner til oppmålingen neste måned?
documents|oil_gas|Please send the latest approved piping drawing for module A.|Send siste godkjente rørtegning for modul A.
documents|manufacturing|Where can I find the operating manual for the milling machine?|Hvor finner jeg bruksanvisningen til fresemaskinen?
documents|engineering|I need a copy of the signed inspection report from last week's test.|Jeg trenger en kopi av den signerte inspeksjonsrapporten fra forrige ukes test.
documents|oil_gas|Please retrieve the material certificate for spool SP-42.|Hent materialsertifikatet for rørseksjon SP-42.
documents|engineering|Which revision of the bracket specification is currently approved? Send that file.|Hvilken revisjon av brakettspesifikasjonen er godkjent nå? Send den filen.
documents|manufacturing|The machine works. I only need the wiring diagram to update our archive.|Maskinen fungerer. Jeg trenger bare koblingsskjemaet for å oppdatere arkivet.
documents|oil_gas|Can document control locate the as-built layout for the separator skid?|Kan dokumentkontroll finne sluttdokumentasjonen med layout for separatorenheten?
documents|engineering|Please provide the original calculation sheet supporting the beam design.|Send det opprinnelige beregningsarket som ligger til grunn for bjelkedesignet.
documents|manufacturing|I am not ordering spare parts. Please forward the manufacturer's parts list as a PDF.|Jeg bestiller ikke reservedeler. Videresend produsentens deleliste som PDF.
documents|oil_gas|We need the maintenance history for valve XV-12 for an audit, with copies of the closed work orders.|Vi trenger vedlikeholdshistorikken for ventil XV-12 til en revisjon, med kopier av avsluttede arbeidsordrer.
documents|engineering|The drawing in the folder is superseded. Can you supply the released version?|Tegningen i mappen er utgått. Kan dere skaffe den frigitte versjonen?
documents|manufacturing|Please locate the calibration certificate for the scale used in goods receiving.|Finn kalibreringssertifikatet for vekten i varemottaket.
it|engineering|I cannot sign in to the engineering document portal. Please restore my access.|Jeg får ikke logget inn på dokumentportalen for engineering. Gjenopprett tilgangen min.
it|manufacturing|The maintenance application crashes when I open a work order. Can IT fix the software?|Vedlikeholdsprogrammet krasjer når jeg åpner en arbeidsordre. Kan IT fikse programvaren?
it|oil_gas|My VPN connection to the office fails with an authentication error.|VPN-tilkoblingen til kontoret feiler med en autentiseringsfeil.
it|engineering|Please unlock my account in the CAD system.|Lås opp kontoen min i CAD-systemet.
it|manufacturing|The pump is fine, but its record will not load in the CMMS. I get an HTTP 500 error.|Pumpen er i orden, men oppføringen åpnes ikke i vedlikeholdssystemet. Jeg får HTTP-feil 500.
it|oil_gas|I know which drawing I need. The portal says permission denied when I download it.|Jeg vet hvilken tegning jeg trenger. Portalen sier at jeg mangler tilgang når jeg laster den ned.
it|engineering|After the desktop update, the simulation program exits before opening a model.|Etter oppdateringen av PC-en avsluttes simuleringsprogrammet før en modell åpnes.
it|manufacturing|The handheld barcode scanner pairs correctly, but the warehouse app refuses my password.|Den håndholdte strekkodeleseren kobles til riktig, men lagerappen avviser passordet mitt.
it|oil_gas|The instrument itself is healthy. The historian dashboard is blank for every asset after the server update.|Selve instrumentet er i orden. Historikkvisningen er tom for alle anleggskomponenter etter serveroppdateringen.
it|engineering|Can the service desk install the approved PDF viewer on my workstation?|Kan brukerstøtte installere den godkjente PDF-leseren på arbeidsstasjonen min?
it|manufacturing|The purchasing page freezes when I submit an order; please investigate the application error.|Innkjøpssiden fryser når jeg sender en bestilling. Undersøk feilen i applikasjonen.
it|engineering|I received the correct report, but the shared drive is unavailable from my laptop.|Jeg har fått riktig rapport, men den delte disken er utilgjengelig fra den bærbare PC-en min.
review|manufacturing|Good morning. Thanks for your help yesterday.|God morgen. Takk for hjelpen i går.
review|oil_gas|Can you help with the thing we discussed? I have no further details yet.|Kan dere hjelpe med det vi snakket om? Jeg har ingen flere detaljer ennå.
review|engineering|I would like to change my holiday dates next month.|Jeg vil endre feriedatoene mine neste måned.
review|manufacturing|Here is the canteen menu for next week.|Her er kantinemenyen for neste uke.
review|oil_gas|Please repair the pump and also reset my portal password. Both requests need separate owners.|Reparer pumpen og tilbakestill også passordet mitt i portalen. Begge forespørslene trenger hver sin ansvarlige.
review|engineering|I need the released drawing and a supplier quote. Please split these into separate requests.|Jeg trenger frigitt tegning og et leverandørtilbud. Del dette i separate forespørsler.
review|manufacturing|The earlier issue is resolved. This is only a thank-you message.|Det tidligere problemet er løst. Dette er bare en takk.
review|oil_gas|Something at the site is wrong, but I do not know whether it is the software or the equipment.|Noe på anlegget er galt, men jeg vet ikke om det er programvaren eller utstyret.
review|engineering|Please reserve a meeting room for the design review.|Reserver et møterom til designgjennomgangen.
review|manufacturing|I found a lost jacket in the changing room. Who owns it?|Jeg fant en gjenglemt jakke i garderoben. Hvem eier den?
review|oil_gas|Today's shift has ended. No equipment faults or requests to report.|Dagens skift er avsluttet. Ingen utstyrsfeil eller forespørsler å melde.
review|engineering|Can you investigate this? The attachment did not come through and there is no description.|Kan dere undersøke dette? Vedlegget kom ikke frem, og det finnes ingen beskrivelse.
"""

DOCUMENTS = """
design|engineering|Bracket drawing DR-120, revision C. Dimensions in millimetres. Hole diameter 12. Material S355.|Braketttegning DR-120, revisjon C. Mål i millimeter. Hulldiameter 12. Materiale S355.
design|oil_gas|Piping specification: all joints in this service shall use the listed flange rating and gasket material.|Rørspesifikasjon: Alle skjøter i denne tjenesten skal bruke angitt flensklasse og pakningsmateriale.
design|manufacturing|Assembly layout. Parts 1 through 8 are positioned as shown. General tolerance plus or minus 0.2 mm.|Sammenstillingstegning. Del 1 til 8 plasseres som vist. Generell toleranse pluss eller minus 0,2 mm.
design|engineering|Section A-A defines the beam geometry. Weld size 6 mm on both sides. Scale 1:5. Released for fabrication.|Snitt A-A definerer bjelkens geometri. Sveisestørrelse 6 mm på begge sider. Målestokk 1:5. Frigitt for fabrikasjon.
design|oil_gas|Line diagram: flow from vessel V-10 through valve XV-12 to pump P-101. Revision 4 changes the bypass arrangement.|Linjeskjema: strømning fra beholder V-10 gjennom ventil XV-12 til pumpe P-101. Revisjon 4 endrer bypass-løsningen.
design|manufacturing|The housing shall be machined from aluminium 6082. Surface finish Ra 1.6 on sealing faces.|Huset skal maskineres i aluminium 6082. Overflateruhet Ra 1,6 på tetningsflatene.
design|engineering|Connector pin allocation: pin 1 supply, pin 2 ground, pin 3 signal. Cable shield terminated at cabinet end.|Kontaktens pinnefordeling: pinne 1 forsyning, pinne 2 jord, pinne 3 signal. Kabelskjerm termineres i skapenden.
design|oil_gas|Design basis for support frame: allowable load 25 kN and maximum deflection 3 mm. Applies to the proposed construction.|Designgrunnlag for støtteramme: tillatt last 25 kN og maksimal nedbøyning 3 mm. Gjelder den planlagte konstruksjonen.
design|manufacturing|Bill of materials for assembly AX-4: two plates, four bushes and eight bolts. This defines the product, with no supplier or prices.|Stykkliste for sammenstilling AX-4: to plater, fire foringer og åtte bolter. Dette definerer produktet, uten leverandør eller priser.
design|engineering|The inspection port shall be located 150 mm above the base. This revision supersedes the earlier dimensional requirement.|Inspeksjonsåpningen skal plasseres 150 mm over sokkelen. Denne revisjonen erstatter det tidligere målkravet.
inspection|manufacturing|Inspection report IR-23. Measured diameter 24.98 mm against requirement 25.00 plus or minus 0.05. Result accepted.|Inspeksjonsrapport IR-23. Målt diameter 24,98 mm mot krav 25,00 pluss eller minus 0,05. Resultat godkjent.
inspection|oil_gas|Pressure test record: held at the specified pressure for thirty minutes. No visible leakage. Witness signed below.|Trykkprøvingsprotokoll: holdt ved angitt trykk i tretti minutter. Ingen synlig lekkasje. Vitne har signert nedenfor.
inspection|engineering|Calibration certificate. Reference value 100, indicated value 100.2. Instrument serial number and technician signature recorded.|Kalibreringssertifikat. Referanseverdi 100, vist verdi 100,2. Instrumentets serienummer og teknikerens signatur er registrert.
inspection|manufacturing|Batch 481 was checked on arrival. Three of fifty pieces had surface scratches. Actual observations recorded on 12 September.|Parti 481 ble kontrollert ved ankomst. Tre av femti deler hadde riper i overflaten. Faktiske observasjoner registrert 12. september.
inspection|oil_gas|Ultrasonic readings at points A, B and C were 8.2, 8.1 and 7.9 mm. The attached map records the measured wall thickness.|Ultralydmålinger i punkt A, B og C var 8,2, 8,1 og 7,9 mm. Vedlagt kart viser målt veggtykkelse.
inspection|engineering|The prototype ran for 200 cycles. Two intermittent resets were observed. This document records completed test results.|Prototypen ble kjørt i 200 sykluser. To sporadiske omstarter ble observert. Dokumentet beskriver resultatene fra gjennomført test.
inspection|manufacturing|Audit of weld W-8: visual examination completed, undercut noted at the upper edge, photograph attached.|Kontroll av sveis W-8: visuell undersøkelse utført, kantsår observert ved øvre kant, fotografi vedlagt.
inspection|oil_gas|Valve XV-12 was stroked five times yesterday. Travel times recorded: 4.1, 4.0, 4.2, 4.1 and 4.0 seconds.|Ventil XV-12 ble kjørt fem ganger i går. Registrerte gangtider: 4,1, 4,0, 4,2, 4,1 og 4,0 sekunder.
inspection|engineering|Against the drawing's 3 mm limit, the measured deflection was 2.4 mm. Test operator and reviewer have signed the results.|Mot tegningens grense på 3 mm var målt nedbøyning 2,4 mm. Testoperatør og kontrollør har signert resultatene.
inspection|manufacturing|Goods receipt quality check: packaging intact, quantity verified, material certificates reviewed. Acceptance recorded by the inspector.|Kvalitetskontroll ved varemottak: emballasje uskadd, antall verifisert, materialsertifikater gjennomgått. Godkjenning registrert av kontrolløren.
purchase|engineering|Purchase order PO-771. Supplier: West Components. Quantity 12 brackets. Unit price NOK 450. Delivery due 15 October.|Innkjøpsordre PO-771. Leverandør: West Components. Antall 12 braketter. Enhetspris 450 kroner. Levering 15. oktober.
purchase|oil_gas|Request for quotation: supply four pressure transmitters. State price, delivery time and commercial terms.|Tilbudsforespørsel: lever fire trykktransmittere. Oppgi pris, leveringstid og kommersielle vilkår.
purchase|manufacturing|Supplier offer: ten conveyor rollers at NOK 800 each. Payment due thirty days after invoice.|Leverandørtilbud: ti transportbåndruller til 800 kroner per stykk. Betaling tretti dager etter faktura.
purchase|engineering|We accept your offer for six machined housings. Total NOK 18,000 excluding tax. Ship to our Oslo workshop.|Vi aksepterer tilbudet på seks maskinerte hus. Totalt 18 000 kroner eksklusive avgifter. Send til verkstedet vårt i Oslo.
purchase|oil_gas|Requisition for approval: two seal kits, estimated total NOK 9,200, preferred vendor listed, delivery address offshore warehouse.|Rekvisisjon til godkjenning: to tetningssett, estimert sum 9 200 kroner, foretrukket leverandør oppgitt, leveringsadresse offshorelager.
purchase|manufacturing|Order acknowledgement: your order for fifty bearings is confirmed. Agreed delivery week 42, net amount NOK 22,500.|Ordrebekreftelse: Bestillingen av femti lagre er bekreftet. Avtalt levering uke 42, nettobeløp 22 500 kroner.
purchase|engineering|Commercial schedule for scanner hire: daily rate NOK 1,500, transport charged separately, offer valid fourteen days.|Kommersielle vilkår for leie av skanner: døgnpris 1 500 kroner, transport faktureres separat, tilbudet gjelder i fjorten dager.
purchase|oil_gas|Please quote for manufacturing the spool to attached drawing SP-42. Include freight and lead time; no installation work is requested.|Gi tilbud på produksjon av rørseksjonen etter vedlagt tegning SP-42. Inkluder frakt og leveringstid. Montering er ikke etterspurt.
purchase|manufacturing|Line items: 20 filter cartridges, 5 drive belts. Charge to cost centre 410. Supplier shall deliver by Friday.|Ordrelinjer: 20 filterpatroner, 5 drivremmer. Belastes koststed 410. Leverandøren skal levere innen fredag.
purchase|engineering|Amendment to PO-388: increase the ordered quantity from eight to twelve. All other agreed prices and delivery terms remain.|Endring av innkjøpsordre PO-388: Øk bestilt antall fra åtte til tolv. Andre avtalte priser og leveringsvilkår beholdes.
work|manufacturing|Work order WO-302. Replace conveyor belt on line 3. Assigned to mechanical workshop. Planned date Monday.|Arbeidsordre WO-302. Skift transportbånd på linje 3. Tildelt mekanisk verksted. Planlagt dato mandag.
work|oil_gas|Maintenance task for pump P-101: inspect the coupling and record findings in this work order. Assigned technician: shift mechanic.|Vedlikeholdsoppgave for pumpe P-101: Undersøk koblingen og registrer funn i denne arbeidsordren. Tildelt tekniker: skiftmekaniker.
work|engineering|Scheduled service instruction for test stand TS-2. Follow the approved service procedure and complete the task checklist.|Planlagt serviceinstruks for teststativ TS-2. Følg godkjent serviceprosedyre og fullfør sjekklisten for oppgaven.
work|manufacturing|Job plan: attend press PR-8 next Tuesday, examine the guard mounting and document any corrective work. Labour allowance two hours.|Jobbplan: Møt ved presse PR-8 neste tirsdag, undersøk festet til vernet og dokumenter eventuelle utbedringer. To arbeidstimer er satt av.
work|oil_gas|Notification converted to WO-918. Scope is to investigate the noisy bearing. Planner has assigned the rotating equipment team.|Melding er omgjort til arbeidsordre WO-918. Omfanget er å undersøke det støyende lageret. Planlegger har tildelt teamet for roterende utstyr.
work|engineering|Workshop task card: repair the damaged bench fixture, use the approved job procedure, record time and parts consumed.|Oppgavekort for verkstedet: Reparer den skadde benkfiksturen, bruk godkjent jobbprosedyre, registrer tidsbruk og forbrukte deler.
work|manufacturing|Weekly preventive maintenance on the packaging unit is due. Technician assignment and task completion fields follow.|Ukentlig forebyggende vedlikehold på pakkeenheten forfaller. Felt for teknikertildeling og fullføring av oppgaven følger.
work|oil_gas|Close-out of WO-501: coupling replaced, four labour hours booked, spare part issued from stores, job completed yesterday.|Avslutning av arbeidsordre WO-501: Kobling skiftet, fire arbeidstimer ført, reservedel tatt fra lager, jobb fullført i går.
work|engineering|Planned intervention: replace the worn guide on the tensile test machine. Responsible group is the laboratory workshop.|Planlagt inngrep: Skift den slitte føringen på strekkprøvemaskinen. Ansvarlig gruppe er laboratorieverkstedet.
work|manufacturing|The attached drawing supports this repair job. Task: restore the damaged bracket to the approved design; record labour against WO-610.|Vedlagt tegning støtter denne reparasjonsjobben. Oppgave: Tilbakestill den skadde braketten til godkjent design. Før timer på arbeidsordre WO-610.
"""


def build_cases():
    result=[]
    for task,body,dev_n in [("routing",ROUTING,4),("documents",DOCUMENTS,3)]:
        counts={}
        for line in body.strip().splitlines():
            label,sector,en,nb=line.split("|")
            index=counts.get(label,0)
            counts[label]=index+1
            family=f"{task}-{label}-{index:02d}"
            for language,text in [("en",en),("nb",nb)]:
                result.append({"id":f"{family}-{language}","family":family,"task":task,"sector":sector,
                    "language":language,"split":"dev" if index<dev_n else "test","gold":label,"text":text})
    return result


if __name__=="__main__":
    cases=build_cases()
    write_json(ROOT/"data/prepared/industry.json",{"description":__doc__,"cases_sha256":fingerprint(cases),"cases":cases})
    print(len(cases),"bilingual records;",len({c["family"] for c in cases}),"scenario families")
