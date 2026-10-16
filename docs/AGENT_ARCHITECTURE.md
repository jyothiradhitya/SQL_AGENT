# Explore-first data agent: architecture for v1

The goal is a chatbot that answers ambiguous database questions and "how do I…" app questions the way
Claude Code did in development: by looking around, checking, and not stopping at the first empty
result. It should not depend on the right table being spoon-fed to it.

Written 2026-10-13 from live runs against `analytical_db` and the app on `:9080`.

---

## 1. How Claude Code answered (the reference behaviour)

There was **no RAG and no semantic KB**. It is one model in a loop with a few general tools:

| Tool | Used for |
|---|---|
| run SQL (read-only) | the live catalog (`information_schema`) and the data itself |
| read / search files | source DDL (`database_snapshot.json`), which holds the real foreign keys |
| fetch URL | the app's HTML and JS bundles |

The loop is **guess → probe → look at the result → adjust → stop only when there is evidence**.

### Example: "what are the CLs present in RA assignment"

1. Search the catalog for table names like `risk`, `assign`, `checklist`, `ra_`. This finds `masters.checklist*`.
   Nothing is named "RA" or "assignment".
2. `checklist_type` looks like a lookup, so **read its rows**. `RA_CFT` = "RA Assessment" appears as a
   *value*, not a table name.
3. Read `checklist`'s columns, see `type_id`, and query the checklists whose type is RA. That gives 2 rows.
4. Check usage in `checklist_entry`. There are 2 entries each, not yet answered.
5. Answer, and say how the terms were interpreted ("RA assignment" read as RA Assessment).

### Example: "department of user admin"

1. Search for columns like `%dep%`/`%user%`. This finds `pc.sys_user` and `masters.party`.
2. The admin's `party.department_id` is **NULL**. **Do not stop on a NULL.**
3. The analytical DB has no FKs, so read the source DDL: `department_id → product_store`.
4. Try the other path, the link table `product_store_user_group`. It is also empty.
5. Answer with evidence: "not assigned; checked X and Y; only 131 of 1,628 users have one".

### Example: "how do I create a user" (UI)

Fetch the app, then its compiled bundles. From those, extract the routes, the menu, the breadcrumbs,
the form fields, the validation messages and the API URLs. Write the steps from that.

**What made it work:**
1. **Exploration tools** instead of one-shot retrieval.
2. **Lookup values are searchable.** Domain words like "RA", "HOD" and "QA" live in rows, not in table names.
3. **Empty or NULL is a signal to keep looking, not an answer.**
4. **Three knowledge sources:** live data, source DDL and app code.
5. **Answers state their evidence and their assumptions.**

---

## 2. Flaws in the current chatbot (verified 2026-10-13)

Live run of `SchemaRetriever.get_relevant_schema()`:

**Q: "what are the cl present in ra assignment"**
```
TABLES: qms.change_control, qms.cc_document, mm.mm_request_history, masters.request, mm.mm_request,
        masters.security_group_permission, masters.equipment, masters.facility, masters.party,
        masters.status, mm.indent, mm.item_issuance, mm.returns, mm.status, qms.item_sub_category
SHORTCUTS: cl = change log ; ra = risk assessment | request approved | remote address
```
No checklist table is retrieved at all, so the SQL agent cannot answer, however good it is.

**Q: "what are the checklists present in RA assessment"** (spelled out in full)
```
TABLES: masters.checklist_entry, masters.checklist_type, masters.checklist, ...   (correct tables)
checklist_type VALUES shown: none   -> RA_CFT / "RA Assessment" never reaches the agent
```

| # | Flaw | Where | Effect |
|---|---|---|---|
| F1 | **Retrieval is a gate, not a hint.** The agent can only use the ≤14 tables retrieval picked, and it has no way to look elsewhere. | `retrieval/pipeline.py`, `sql_agent.py` rules ("Use ONLY tables… shown") | One bad embedding match makes the question unanswerable (the RA question). |
| F2 | **Shortcut miner only knows multi-word initials.** It misses truncations (`CL`=checklist) and acronyms that appear *literally* in values (`RA_CFT`, "RA Assessment"). | `retrieval/shortcuts.py` | `cl` becomes "change log" and `ra` becomes "remote address": confidently wrong. |
| F3 | **Lookup VALUES depend on `table_role`.** `checklist_type` has its own `type_id`/`status_id`, so it is classed "business entity" and its rows are never loaded. | `kb.py add_cross_table_fields`, `sample_values.py` | The words users actually use ("RA", "HOD", "QA closing") are invisible. |
| F4 | **The 14-table cap trims lookups first** ("trimming lookups/identities to 0"). Bridges to stray anchors are kept. | `pipeline.py` cap logic | The same root cause as the CC-status bug (`masters.status` instead of `qms.status`). |
| F5 | **First `SQL_SUCCESS` ends the agent.** A NULL/empty column in the answer is never investigated. | `sql_agent.py` loop | "department = NULL" would be reported without checking the other paths. |
| F6 | **No catalog tools.** The agent can't list columns, follow FKs in both directions, or peek at rows unless retrieval happened to include the table. | MCP server has only `get_relevant_schema`, `run_sql` | It can't do steps 1–3 of the reference behaviour. |
| F7 | **The answer has no evidence or assumptions.** Interpretations ("RA assignment → RA Assessment") are silent. | orchestrator prompt | Wrong guesses look like facts. |
| F8 | **No app (UI) knowledge.** | – | "How do I…" questions can't be answered. |

**RAG isn't the problem. Using RAG as the only door is.** Keep it as a fast first guess, and let the
agent explore when the guess doesn't fit.

---

## 3. Target architecture

```
User
 │
Orchestrator (chat, history, final wording, evidence line)
 ├── data_agent(question)          ── explore-first SQL agent (§3.1)
 │     tools: find_tables, describe_table, find_values, get_relevant_schema (RAG hint), run_sql
 │
 └── app_guide(question)           ── search over app knowledge (§3.2)
       tools: search_app_screens
                        ▲
     offline builders:  kb.py (DB)   ui_kb.py (app)   bridge (UI words → tables)
```

### 3.1 Data agent: explore first

**Tools.** All are read-only, run through the MCP server, and are deterministic: no LLM inside a tool.

| Tool | Input | Returns | Source |
|---|---|---|---|
| `find_tables(terms)` | words, e.g. `["checklist","ra","assign"]` | table/column hits ranked by match on table name, column name, KB name/aliases, **and lookup values** | catalog + KB + value index |
| `find_values(term)` | `"RA"`, `"HOD"`, `"approved"` | `table.column = value (id)` matches, using exact, prefix, acronym (`RA_CFT`→RA) and fuzzy matching | value index over **all** small text columns (§3.3) |
| `describe_table(table)` | `masters.checklist` | columns + types, PK, FKs out **and in** (from source DDL), row count, 3 sample rows (sensitive columns masked), large-object flags | catalog + snapshot + live |
| `get_relevant_schema(q)` | question | existing RAG context, **as a hint** | existing pipeline |
| `run_sql(sql)` | SELECT | rows + `NULL_COLUMNS:` note + existing empty-result check | existing executor |

**Loop rules** (prompt plus a small amount of code):
1. Start with `get_relevant_schema`. If the tables plainly don't fit the question's nouns, or a
   term is unknown, use `find_values` / `find_tables`. **Never refuse because a table was not in the hint.**
2. Resolve every domain word to a *value or a table* before writing SQL ("RA" → `checklist_type.name='RA_CFT'`).
3. After `run_sql`:
   - Success with real values: stop.
   - A requested column is all NULL, or there are 0 rows: **one investigation round**. Use
     `describe_table` on the FK target or the incoming link tables, try the alternative path, then stop.
4. Budget: max 10 tool calls / 6 SQL runs. When it runs out, return the best evidence found so far.
5. Output a block the orchestrator can phrase:
   ```
   RESULT: <rows>
   INTERPRETATION: "cl"=checklist (masters.checklist); "ra assignment"=checklist_type RA_CFT "RA Assessment"
   EVIDENCE: checklist_type has 2 RA rows; checklist_entry has 2 entries each, unanswered
   QUERY_RUN: ...
   ```

**Remove from the SQL prompt:** "Use ONLY tables and columns shown in the schema you were given".
Replace it with: "use only tables/columns you have *seen* through any tool". The validator still
rejects unknown tables and columns, so safety doesn't change.

### 3.2 App guide: UI knowledge

**Offline builder `ui_kb.py`** (same pattern as `kb.py`: incremental, hash per screen):
1. `GET APP_URL` → `runtime.*.js` → the chunk map → download all chunks.
2. Extract the following (Angular compiled output; regexes were validated on `:9080`):
   - **Menu:** `{name, url, authorities, submenu}` arrays (e.g. User Management → Users List).
   - **Routes:** `{path, component | loadChildren}` → full URLs (`/masters/user-mangement/list/create`).
   - **Per component** (`selectors:[["sys-create-user"]]`, segment up to the next component):
     - `formControlName`s, labels (`EFF(n,"Employee ID")`) and placeholders
     - validation messages ("Email is required"), buttons, toasts ("created successfully")
     - breadcrumbs (`setBreadcrumb([...])`) and permission checks (`authorities.includes("USER_MANAGEMENT_CREATE")`)
   - **Services:** `method(){return this.http.post(this.masterUrl+"api/parties/party/create-employee")}`
     → method → (verb, URL, backend = masters/qms/dms/lms). Component calls `this.x.method(` are then linked to the API.
3. An LLM writes 2–3 sentences per screen: what it is for and the main steps. These are embedded.
4. Output `data/ui_kb.json`: `{url, menu_path, breadcrumb, permissions, fields[{name,label,required,validation}], buttons, apis[], description}`.

**Runtime tool** `search_app_screens(question)` returns the top 3 screens. The orchestrator answers with
the menu path, the fields (required ones marked), the required permission, and alternatives
(request/approval flow, self-registration).

v1 scope: **Angular apps** (`:9080`). React (`:9090`) needs a different extractor, so it is v2.

### 3.3 Knowledge stores, and the bridge between them

| Store | Built by | Fixes |
|---|---|---|
| **Value index**: every text column with ≤500 distinct short values, in **every** table (not only `table_role=lookup`), plus acronyms derived from values (`RA_CFT`→RA, "Quality Assurance"→QA) | `build_index` (offline) | F2, F3 |
| **FK graph**: both directions, from the source DDL | existing `fk_graph.json` | F6 |
| **UI→DB bridge**: screen label + API resource → table. Example: the "Department" field on the create-employee screen posts to `api/parties` (`party.department_id` → `product_store`), so "department" is an alias of `product_store`. | `ui_kb.py` post-step | ambiguity in table names |
| **Shortcut miner v2**: also truncations (`CL`→checklist: the word starts with these letters) and literal tokens in values (`RA`) | `shortcuts.py` | F2 |

### 3.4 Retrieval fixes (even before the agent change)
- **Cap:** never trim a lookup that an anchor's FK points to directly, and never trim a table whose
  *values* matched a question term. Trim multi-hop bridges first (F4).
- When a question term matches a **value** (value index), add that value's table as an anchor.

---

## 4. Safety (unchanged and required)
- **Read-only:** read-only transactions, `statement_timeout`, row limits, sensitive-column blocklist.
- **Tool output masking:** `describe_table` sample rows mask blocked columns.
- **UI knowledge:** describes screens and the permissions they need. It never performs actions.
- **No RBAC yet:** the bot can still describe data the user couldn't see in the app (known gap).

---

## 5. Build order (each step is testable on its own)

| Step | Change | Files | Done when |
|---|---|---|---|
| 1 | Value index for all small text columns, plus value acronyms | `retrieval/sample_values.py`, `build_index.py` | `find_values("RA")` → `checklist_type RA_CFT` |
| 2 | `find_tables`, `find_values`, `describe_table` MCP tools | `mcp_server/server.py`, new `retrieval/catalog.py` | Tools return the expected rows for the 3 examples |
| 3 | SQL agent: explore-first prompt, NULL/empty investigation round, budget, INTERPRETATION/EVIDENCE output | `agents/sql_agent.py`, `sql/formatting.py` | RA question and admin-department question answered correctly |
| 4 | Retrieval cap fix + value-matched anchors | `retrieval/pipeline.py` | CC status uses `qms.status` |
| 5 | Shortcut miner v2 | `retrieval/shortcuts.py` | `cl`→checklist, `ra`→RA Assessment |
| 6 | `ui_kb.py` builder + `search_app_screens` + `app_guide` tool | new `ui_knowledge/`, `orchestrator.py` | "how do I create a user" = the menu path + fields + permission |
| 7 | UI→DB bridge aliases into the KB | `ui_knowledge/bridge.py`, `kb.py` | "department" → `product_store` without a KB alias |
| 8 | Test set: 30 questions (15 data, 15 how-to), live runs | `experiments/` | Report pass rate per step |

**Model note:** exploration needs more reasoning than one-shot SQL. If `gpt-oss:20b` loops or
stops early in step 3, run the data agent on `gpt-oss:120b-cloud` and keep 20b for the cheap parts.

## 6. Acceptance questions (must pass for v1)
| Question | Expected |
|---|---|
| what are the cl present in ra assignment | 2 checklists: 0001 "Is amendment required or not?", 0002 "Is customer notification required?" (both approved); says it read "RA" as RA Assessment |
| department of user admin | Not assigned (party.department_id NULL, no product_store_user_group rows) |
| how many change controls and their statuses | 5: CC_HOD_RETURNED 2, CC_QA_ASSESSED 2, CC_POST_IMP_QA_COMPLETED 1 |
| how do I create a user | Masters → User Management → Users List → Create; fields (Employee ID, Username, Email…, Remarks required); needs USER_MANAGEMENT_CREATE, USER_MANAGEMENT_ADMIN or ROLE_ADMIN |
