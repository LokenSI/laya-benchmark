SCENARIO_DESCRIPTIONS = {
    'alarm': 'set or change alarms', 'audio': 'volume and sound settings',
    'calendar': 'appointments and reminders', 'cooking': 'recipes and cooking',
    'datetime': 'date and time', 'email': 'read or send email',
    'general': 'assistant conversation and feedback', 'iot': 'smart home devices',
    'lists': 'manage lists', 'music': 'music preferences and information',
    'news': 'news headlines', 'play': 'play media or games',
    'qa': 'factual questions and calculations', 'recommendation': 'suggest places and events',
    'social': 'social media messages', 'takeaway': 'order food delivery',
    'transport': 'travel and traffic', 'weather': 'weather forecasts',
}

def question(suite):
    if suite["task"] == "scenario":
        return {"decision": {"type": "choice", "instructions": "Which assistant service should handle this request?",
                "criteria": {key: SCENARIO_DESCRIPTIONS[key] for key in suite["labels"]}}}
    return {"decision": {"type": "choice", "instructions": "What sentiment is expressed in this review sentence?",
            "criteria": {"negative": "criticism or dissatisfaction", "neutral": "no positive or negative sentiment",
                         "positive": "praise or satisfaction"}}}

def support_questions(norwegian=False):
    if norwegian:
        return {
            "route": {"type": "choice", "instructions": "Hvilket team bør behandle kundens forespørsel?",
                "criteria": {"billing": "faktura, betaling og refusjon", "technical": "feil og driftsproblemer",
                             "sales": "priser og kjøp", "account": "innlogging og kontotilgang", "other": "andre henvendelser"}},
            "refund": {"type": "noul", "instructions": "Ber kunden uttrykkelig om å få penger tilbake?"},
            "refund_choice": {"type": "choice", "instructions": "Ber kunden uttrykkelig om å få penger tilbake?",
                "criteria": {"A": "nei, kunden ber ikke om refusjon", "B": "ja, kunden ber om refusjon"}},
            "urgency": {"type": "score", "instructions": "Hvor mye haster denne henvendelsen?",
                "criteria": ["ingen tidsfrist eller driftsstans", "konkret frist i dag, men ingen full driftsstans", "full driftsstans for alle brukere nå"]}}
    return {
        "route": {"type": "choice", "instructions": "Which team should handle the customer's request?",
            "criteria": {"billing": "invoices, payments and refunds", "technical": "bugs and outages",
                         "sales": "pricing and purchases", "account": "login and account access", "other": "other requests"}},
        "refund": {"type": "noul", "instructions": "Does the customer explicitly request money back?"},
        "refund_choice": {"type": "choice", "instructions": "Does the customer explicitly request money back?",
            "criteria": {"A": "no, the customer does not request a refund", "B": "yes, the customer requests a refund"}},
        "urgency": {"type": "score", "instructions": "How urgent is this request?",
            "criteria": ["no deadline or service outage", "explicit deadline today but no complete outage", "complete service outage for all users right now"]}}
