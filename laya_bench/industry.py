"""Frozen workflows and a deliberately simple, transparent keyword comparator."""
import re

ROUTES={
    "maintenance":"physical equipment faults, inspections or repairs by a maintenance technician",
    "procurement":"buying, hiring, replenishing parts, supplier prices or commercial offers",
    "documents":"finding, supplying or checking the revision of technical documents or records",
    "it":"software faults, computer accounts, permissions, networks or application installation",
    "review":"unclear request, multiple different requests, or none of the above"}
ROUTES_NB={
    "maintenance":"fysiske utstyrsfeil, inspeksjon eller reparasjon utført av vedlikeholdstekniker",
    "procurement":"innkjøp, leie, påfyll av deler, leverandørpriser eller kommersielle tilbud",
    "documents":"finne, sende eller kontrollere revisjon av tekniske dokumenter og historikk",
    "it":"programvarefeil, datakontoer, tilganger, nettverk eller installasjon av applikasjoner",
    "review":"uklar forespørsel, flere forskjellige forespørsler, eller ingen av de andre kategoriene"}
DOCS={
    "design":"drawing, specification, design basis or bill of materials defining what to build",
    "inspection":"inspection, test or calibration record describing measured observations and results",
    "purchase":"purchase order, requisition, quotation or commercial agreement to obtain goods or services",
    "work":"maintenance work order, job plan, task card or completed maintenance job record"}
DOCS_NB={
    "design":"tegning, spesifikasjon, designgrunnlag eller stykkliste som definerer hva som skal bygges",
    "inspection":"inspeksjons-, test- eller kalibreringsrapport som beskriver målinger og resultater",
    "purchase":"innkjøpsordre, rekvisisjon, tilbud eller kommersiell avtale om varer eller tjenester",
    "work":"arbeidsordre, jobbplan, oppgavekort eller registrering av utført vedlikeholdsarbeid"}


def question(task,language,variant):
    localized=language=="nb" and variant=="contextual"
    criteria=(ROUTES_NB if localized else ROUTES) if task=="routing" else (DOCS_NB if localized else DOCS)
    if task=="routing":
        instructions=("Hvilken gruppe skal behandle forespørselen? Velg etter det avsenderen ber om, ikke utstyr som bare nevnes. Flere ulike forespørsler eller utilstrekkelig informasjon går til review."
            if localized else "Which team should handle this request?")
        if language=="en" and variant=="contextual":
            instructions="Route by the action requested, not equipment merely mentioned. Choose review for unrelated, unclear or multiple requests. Which team should handle it?"
    else:
        instructions=("Hva slags dokument er dette utdraget fra? Velg dokumentets hovedformål, ikke andre dokumenter det henviser til." if localized else "What type of engineering document is this excerpt from?")
        if language=="en" and variant=="contextual":
            instructions="Classify the main purpose of this document excerpt, not a different document referenced inside it. Which document type is it?"
    return {"decision":{"type":"choice","instructions":instructions,"criteria":dict(criteria)}}


KEYWORDS={
    "routing":{
        "maintenance":["repair","leak","vibrat","bearing","technician","inspect","reparer","lekker","vibrer","lageret","tekniker","undersøk"],
        "procurement":["order","quot","purchase","price","buy","hire","supplier","bestill","tilbud","innkjøp","pris","kjøp","leie","leverandør"],
        "documents":["drawing","manual","certificate","report","specification","revision","tegning","bruksanvisning","sertifikat","rapport","spesifikasjon","revisjon"],
        "it":["password","log in","sign in","account","vpn","software","app","http","permission","passord","logget inn","konto","program","tilgang"]},
    "documents":{
        "design":["drawing","specification","design","dimensions","tolerance","tegning","spesifikasjon","toleranse","stykkliste"],
        "inspection":["inspection","measured","test record","calibration","results","observed","inspeksjon","målt","kalibrering","resultat","observert"],
        "purchase":["purchase","quotation","supplier","price","payment","po-","innkjøp","tilbud","leverandør","pris","betaling"],
        "work":["work order","wo-","maintenance","task card","job plan","labour","arbeidsordre","vedlikehold","oppgavekort","jobbplan","arbeidstime"]}}


def keyword_route(text,task):
    lower=text.casefold()
    scores={label:sum(term in lower for term in terms) for label,terms in KEYWORDS[task].items()}
    high=max(scores.values())
    winners=[label for label,value in scores.items() if value==high]
    return winners[0] if high and len(winners)==1 else "review"


def asset_tags(text):
    # Copy exact identifiers, never infer a tag from the meaning of the sentence.
    return list(dict.fromkeys(re.findall(r"\b(?:P|C|V|XV|PR|TS|SP)-\d+\b",text)))


def validate_input(payload):
    if not isinstance(payload,dict):
        raise ValueError("Enter a request or document excerpt.")
    text=payload.get("text")
    if not isinstance(text,str) or not text.strip() or len(text)>4000:
        raise ValueError("Enter between 1 and 4,000 characters.")
    if payload.get("task") not in ("routing","documents") or payload.get("language") not in ("en","nb"):
        raise ValueError("Choose a supported workflow and language.")
    return text.strip(),payload["task"],payload["language"]
