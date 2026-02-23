# Analisi Criticita del Progetto Maki

## Sintesi
Il progetto e una libreria client per Ollama con un layer multi-agent. La base e chiara, ma ci sono criticita funzionali e di robustezza che possono causare crash, comportamento non deterministico e mismatch tra documentazione e runtime.

## Criticita (ordinate per severita)

### 1) Compatibilita Python incoerente (CRITICA)
- Evidenza:
  - Uso di `StrEnum` in `maki/urls.py:1` (richiede Python 3.11+).
  - README dichiara `Python 3.6+` in `README.md:197`.
- Impatto:
  - ImportError su Python 3.10 e inferiori, con test non avviabili.
- Verifica:
  - `python3 -m unittest discover -s maki/test -v` fallisce con `ImportError: cannot import name 'StrEnum'`.
- Fix consigliato:
  - Sostituire `StrEnum` con `Enum` + valori stringa, oppure alzare requisito minimo a Python 3.11 in docs/metadata.

### 2) Payload globale mutabile condiviso tra richieste (CRITICA)
- Evidenza:
  - Oggetto globale: `maki/llm_objects/ollama_payload.py:1`.
  - Modifica in-place in `maki/maki.py:43`.
  - Campi opzionali aggiunti senza reset (`options`, `images`) in `maki/maki.py:47` e `maki/maki.py:50`.
- Impatto:
  - Possibile contaminazione tra chiamate (una richiesta puo ereditare `images`/`options` da una precedente).
  - Race condition in uso concorrente.
- Fix consigliato:
  - Creare un nuovo dizionario payload per ogni richiesta (deep copy o factory function).

### 3) Composizione URL fragile e incoerente con gli esempi (ALTA)
- Evidenza:
  - Template URL non impone schema: `maki/urls.py:7`.
  - `compose_url` concatena direttamente: `maki/utils.py:20`.
  - Esempi README usano `url="localhost"` senza `http://` (`README.md:28`).
- Impatto:
  - Richieste HTTP potenzialmente invalide (`MissingSchema`) se l'utente segue la README.
- Fix consigliato:
  - Normalizzare `url` aggiungendo schema di default (`http://`) quando assente.
  - Allineare README ed esempi.

### 4) HTTP senza timeout e gestione errori troppo generica (ALTA)
- Evidenza:
  - `requests.post(...)` senza timeout in `maki/connector.py:23` e `maki/connector.py:46`.
  - Rilancio con `Exception` generica in `maki/connector.py:28`, `maki/connector.py:30`, `maki/connector.py:50`.
- Impatto:
  - Possibile blocco indefinito in rete lenta/non raggiungibile.
  - Perdita di tipizzazione eccezioni e diagnosi piu difficile.
- Fix consigliato:
  - Introdurre timeout configurabile e propagare eccezioni specifiche o custom exception dedicate.

### 5) Parsing risposta rigido su chiave `response` (MEDIA)
- Evidenza:
  - Accesso diretto `jsonify["response"]` in `maki/connector.py:26`.
- Impatto:
  - `KeyError` se il payload di errore/risposta cambia formato.
- Fix consigliato:
  - Validare schema e gestire fallback/error payload in modo esplicito.

### 6) `coordinate_agents` non usa `coordination_prompt` e puo sovrascrivere risultati (MEDIA)
- Evidenza:
  - Parametro non usato: `maki/agents.py:157`.
  - Dizionario indicizzato con testo task: `results[task] = result` in `maki/agents.py:180`.
- Impatto:
  - Prompt di coordinamento ignorato.
  - Task con stesso testo si sovrascrivono.
- Fix consigliato:
  - Usare `coordination_prompt` nella costruzione del prompt.
  - Usare ID step univoco come chiave (indice o campo `name`).

### 7) `collaborative_task` e collaborazione "simulata" (MEDIA)
- Evidenza:
  - Singola chiamata `self.maki.request(prompt)` in `maki/agents.py:209`.
  - Nessuna raccolta output dagli agenti elencati.
- Impatto:
  - La feature non realizza vera collaborazione multi-agent, solo role-play in un unico prompt.
- Fix consigliato:
  - Eseguire task per ciascun agente e fare sintesi finale con coordinator.

### 8) Stato/attributi non usati (BASSA)
- Evidenza:
  - `task_queue` definita ma mai usata in `maki/agents.py:86`.
  - Memoria agente (`remember/recall`) non integrata nella composizione prompt (`maki/agents.py:61`, `maki/agents.py:65` vs `maki/agents.py:48`).
- Impatto:
  - Codice morto e funzionalita dichiarate ma non operative.
- Fix consigliato:
  - Rimuovere attributi non usati o integrarli nel workflow.

### 9) Documentazione installazione incompleta (BASSA)
- Evidenza:
  - README indica solo `pip install requests` (`README.md:17`).
- Impatto:
  - Onboarding ambiguo per usare la libreria come pacchetto.
- Fix consigliato:
  - Aggiungere istruzioni complete (install locale, editable mode, eventuale packaging).

## Priorita di intervento consigliata
1. Risolvere compatibilita Python (`StrEnum`) e allineare requisiti/documentazione.
2. Eliminare il payload globale condiviso e rendere le richieste stateless.
3. Mettere timeout + gestione errori robusta nel connector.
4. Correggere composizione URL e esempi README.
5. Rifinire reale orchestrazione multi-agent (`coordinate_agents`/`collaborative_task`).

## Miglioramenti consigliati (pratici)

### A) Hardening API client
- Obiettivo:
  - Rendere il client affidabile in ambienti reali.
- Interventi:
  - Timeout configurabile (`connect` + `read`) in `Connector`.
  - Retry con backoff solo su errori transienti (5xx, timeout).
  - Eccezioni custom (`MakiConnectionError`, `MakiResponseError`) al posto di `Exception` generica.
  - Validazione risposta JSON con fallback esplicito.

### B) Coerenza configurazione e DX
- Obiettivo:
  - Ridurre errori di setup lato utente.
- Interventi:
  - Normalizzare URL in input (`localhost` -> `http://localhost`).
  - Esporre `base_url` unico nel costruttore, mantenendo retrocompatibilita.
  - Allineare README, esempi e requisiti Python reali.
  - Aggiungere quickstart minimo realmente eseguibile.

### C) Architettura payload e stato
- Obiettivo:
  - Eliminare side effect nascosti.
- Interventi:
  - Costruire payload immutabile per singola request.
  - Introdurre metodo `_build_payload(...)` puro.
  - Evitare stato condiviso globale in `maki/llm_objects/ollama_payload.py`.

### D) Qualita multi-agent
- Obiettivo:
  - Rendere la collaborazione effettiva, non simulata.
- Interventi:
  - `collaborative_task`: eseguire task per ogni agente e sintetizzare nel passo finale.
  - `coordinate_agents`: usare `coordination_prompt` realmente.
  - Definire schema risultato stabile per step (`id`, `agent`, `input`, `output`, `status`).
  - Gestire failure parziali (un agente fallisce, workflow prosegue con stato degradato).

### E) Testing e CI
- Obiettivo:
  - Evitare regressioni e mismatch ambiente.
- Interventi:
  - Matrice test su Python supportati.
  - Test su casi di errore API (JSON invalido, timeout, chiavi mancanti).
  - Test di non contaminazione payload tra richieste consecutive.
  - Lint/type-check minimi.

## Errori di flusso (as-is)

### 1) Flusso bootstrap/test bloccato in import
- Flusso atteso:
  - Import package -> esecuzione test.
- Flusso reale:
  - Import di `maki/urls.py` fallisce su Python < 3.11 per `StrEnum`.
- Effetto:
  - La suite non parte, quindi nessuna validazione funzionale.

### 2) Flusso request text con possibile URL invalido
- Flusso atteso:
  - `Maki.request` -> URL valido -> `requests.post`.
- Flusso reale:
  - Se utente usa README (`url="localhost"`), URL finale puo essere senza schema.
- Effetto:
  - Errore di connessione/`MissingSchema` prima di raggiungere Ollama.

### 3) Flusso request image con side effect sulla request successiva
- Flusso atteso:
  - Request con immagini isolata, request successiva pulita.
- Flusso reale:
  - Campo `images` puo rimanere nel payload globale mutato.
- Effetto:
  - Richieste successive possono inviare dati non previsti.

### 4) Flusso temperatura non deterministico tra chiamate
- Flusso atteso:
  - Ogni chiamata riflette solo la configurazione corrente.
- Flusso reale:
  - `options` viene impostato solo se temperatura truthy; con payload globale puo restare residuo.
- Effetto:
  - Parametri request non sempre allineati allo stato desiderato.

### 5) Flusso coordinate_agents perde dati in caso di task duplicati
- Flusso atteso:
  - Tutti i task restituiti in output.
- Flusso reale:
  - Usa il testo task come chiave dizionario (`results[task]`), quindi overwrite su task uguali.
- Effetto:
  - Risultati mancanti/sovrascritti.

### 6) Flusso collaborative_task non interroga davvero gli agenti
- Flusso atteso:
  - Ogni agente produce output, poi coordinamento.
- Flusso reale:
  - Una sola chiamata al modello centrale con lista nomi agenti nel prompt.
- Effetto:
  - Collaborazione apparente, non verificabile e non tracciabile per agente.

### 7) Flusso error handling opaco
- Flusso atteso:
  - Errore classificato e propagato con contesto.
- Flusso reale:
  - Wrapping in `Exception` generica.
- Effetto:
  - Difficile distinguere rete/HTTP/parsing e applicare recovery mirata.
