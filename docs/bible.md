# The bible

The single source of truth for how this system works. One file, chaptered. A chapter ID is the
slug after the `##` (for example `architecture/system-context`), and the chapter map in root
[`CLAUDE.md`](../CLAUDE.md) points at it; link to a chapter as `docs/bible.md#architecturesystem-context`.

**How to use it.** Read the chapter the work touches, never the whole file. When you learn
something load-bearing that a future reader would otherwise have to re-derive from code, write it
into its chapter in the same session. When a chapter contradicts the code, the code wins and the
chapter gets fixed on the spot.

**What does not go here.** Vocabulary goes in [`CONTEXT.md`](../CONTEXT.md). Decisions that were
hard to reverse go in [`docs/adr/`](adr/). Status of ongoing work goes on its GitHub issue. The
bible holds how the thing works, not what we are doing about it.

**Budget.** 900 words and 8 sections per chapter; `.claude/hooks/local/bible_guard.py` blocks an
edit that grows a chapter past either without a line starting `Budget:` saying why. A drawn
module's boxes are each claimed by one piece of text here carrying its ◈ marker, and
`docs/diagrams/tools/linkmap.py` fails on a box nothing claims.

---

## architecture/system-context

What the harness is: one repository that holds a small company's whole working system, and the
loop a person and Claude Code use to change it. Push to `main` is the deploy.

### The pieces

```
your system: Claude Code in this repository
  GitHub: main, issues, the board
    the server: one Scaleway VM, the compose stack behind Caddy
      Nextcloud: files, calendar, the single sign-on
      Notulen: meetings in, minutes out
      the assistant: Signal in and out
      observability: Prometheus, Alertmanager, probes
    Scaleway: object storage, Secret Manager, optional GPU cluster
```

Nothing deploys from your system.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=laptop) Work lands on `main`, the server pulls it, the deploy gate tests
the commit against its own tree, and only a green commit is swapped in. A red one holds the old
version, and with `EJECT_ENABLED=1` it is reverted on `main` and handed back to its author.

### Who talks to it

| Who | Through | Reaches |
|---|---|---|
| The owner and the team | a browser, one Nextcloud login | files, calendar, Notulen, gated pages |
| The owner and the team | Signal, both ways | the assistant, which asks, pages and briefs back |
| A session | git and `gh` | the repository, issues, the board[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=github) |
| The public | HTTPS | the website at the root domain |

Every name is a subdomain of `DOMAIN`. Caddy terminates HTTPS for all of them. A gated name sends
an anonymous browser through oauth2-proxy to Nextcloud, which is the identity provider, so one
login with its 2FA code opens everything.

### Where configuration lives

| Place | Holds | Written by |
|---|---|---|
| `.claude/project.json` | the GitHub side: repo, board, people | `.claude/scripts/setup-github.py` |
| `config/harness.env` | non-secret settings | the owner, from `config/harness.env.example` |
| the vault | the full environment, secrets included | `infra/secrets/push-env.sh` |

Nothing else is read. The vault is one Scaleway Secret Manager secret (`VAULT_SECRET_NAME`); the
server pulls it every deploy cycle to `$HARNESS_ROOT/harness.env`, mode 600. Scaleway API keys
stay in the laptop's `scw` config. The `onboard` skill takes a fresh laptop to a running server
and fills all three.

### What is not in it

No business content, no customer data and no person's name ships with the repository. The
assistant starts empty and is wired to the owner's own Signal number, calendar and file cloud
during onboarding. Minutes, memory notes and the people list are written into the owner's own
private copy as they happen.

## apps/assistant

### What runs

| Container | Image | Job |
|---|---|---|
| `assistant-gateway` | upstream OpenClaw, pinned | holds the Signal conversation, the sessions and its own memory |
| `assistant` | built from `apps/assistant/Dockerfile` | the door: alert triage, the notify pen, patrol, the MCP tool door |
| `signal-cli` | signal-cli-rest-api, pinned | the Signal account both of them use |
| `assistant-whisper` | speaches, profile `voice` | turns a voice note into text |

The gateway is someone else's software, run unmodified. It answers people. The door is this repository's code, and it decides what reaches a phone.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=assistant) Neither container is reachable from the internet: the gateway and the door publish only on the server's loopback, and Alertmanager, Prometheus and the gateway reach the door by its name, `assistant`, on the compose network. The door keeps its memory in one SQLite file on a named volume and runs a single worker, because that file has one writer. In the cluster the same four are Deployments: the gateway alone in namespace `assistant-gateway`, the rest in `assistant`, where a network policy lets only the door and the gateway reach signal-cli.

### How an alert reaches Signal

```
Alertmanager
  -> POST /webhook/alertmanager    (bearer OPENCLAW_WEBHOOK_SECRET)
  -> triage                        (the model reads, then pages or stays silent)
  -> the notify pen                (pages held and batched)
  -> the flush                     (quiet hours, mutes, one message)
  -> signal-cli /v2/send
```

Triage gives the model the alert and its read-only tools, and asks for one decision: page, enrich or stay silent. A page then waits in the notify pen, so a burst of related alerts leaves as one message. Night-time warnings wait for the morning, and a critical page leaves at any hour.

Every sentence crosses the claim guards before it is sent: a sentence about live state that no tool read this pass is dropped, and so is a command for the reader to type. Every send is written to the state database, and the chat relay copies it into the recipient's own gateway session, so a reply in chat has the context it answers.

### The tool door

The gateway calls the door at `http://assistant:8000/mcp` for every tool that touches the team's systems. The read tools are the calendars, the board, Prometheus, Loki when `LOKI_URL` is set, the ops database, the bible, deploy status, the repository's documents and the assistant's own memory. The write tools are the calendar, video calls, the board, a message to a teammate and the assistant's memory.

Each write has its own switch, needs `requested_by` naming the person who asked, and is recorded in the action log before it runs. A tool whose configuration is missing is not offered at all. A tool that ran and failed says so, and never comes back as an empty answer.

### The workspace

| File | Rendered from |
|---|---|
| `AGENTS.md` | the principles the owner taught and the tool list |
| `SOUL.md` | the repository's `.claude/soul.md` |
| `USER.md` | the people registry and each person's own rules |
| `IDENTITY.md` | the assistant's name |

The gateway builds its system prompt from these four files. `apply.sh` renders them from the live door on every deploy. The same pass applies the gateway's config fragment and declares two scheduled reports, a weekday brief and a Monday advisor, at the times the owner set. Both reports leave through `message_teammate`, where the word cap is a number in code. A file the assistant edits by hand is gone at the next deploy, which is the point: the doctrine has one home.

### What the owner fills

| Where | Keys |
|---|---|
| `config/harness.env` | `ASSISTANT_NAME`, `ASSISTANT_TIMEZONE`, `ASSISTANT_BRIEF_TIME`, `ASSISTANT_ADVISOR_TIME`, the write switches, `AGENDA_BASE_URL`, `AGENDA_CALENDARS`, `TEAM_CALENDAR` |
| the vault | `SIGNAL_BOT_NUMBER`, `SIGNAL_ALLOWLIST`, `SIGNAL_PEOPLE`, `SIGNAL_RECIPIENT`, `AGENDA_USER`, `AGENDA_APP_PASSWORD`, `GITHUB_ASSISTANT_TOKEN`, `OPENCLAW_WEBHOOK_SECRET`, `OPENCLAW_GATEWAY_TOKEN` |
| `.claude/project.json` | the repository it works on, the people's GitHub logins, the board's areas |

The assistant ships empty: no people, no memory, no knowledge beyond one example note. Before the first start, link the bot number to signal-cli once with `SIGNAL_CLI_MODE=native`, then set it back.

## apps/notulen

### The recording path

```
phone browser -> 5 s chunks -> /api/upload/chunk -> /tmp/notulen/<job>/audio.webm
stop -> /api/upload/finalize -> stitch -> loudness boost -> S3 -> process_job
```

The recorder is a single page at `notulen.$DOMAIN`, behind the Nextcloud single sign-on.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=notulen) Pressing record opens a row in `notulen_jobs` with status `recording`. From then on the browser sends five seconds of audio at a time. Each piece is kept in the phone's IndexedDB until the server confirms it, so a locked screen or a closed tab loses nothing. The next visit offers "Resume upload".

The server appends the pieces to one file on a volume, so a redeploy does not wipe a meeting in progress; in the cluster that volume is the claim `notulen-sessions` in namespace `notulen-app`. Gates refuse a chunk out of order (409 asks the browser to re-send from the gap), a chunk over 10 MB, a recording under 10 KB, and a third concurrent job.

When the microphone dies mid-meeting, the page beeps, vibrates and keeps asking for it; recording resumes into the same session. At finalize the server stitches the pieces and boosts a near-silent recording. The audio goes to S3 before any check that could end the session, so every failed job can be retried.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=storage)

### Transcription and minutes

| Tier | When | Transcribes with | Speaker labels |
|---|---|---|---|
| GPU | `GPU_ENABLED=1` and `KUBECONFIG_B64` set | whisper-large-v3 in a Kubernetes Job | yes, with `HF_TOKEN` |
| CPU | otherwise, or when GPU dispatch fails | whisper `medium` in the dashboard | no |

Either tier ends the same way. Under 50 words is `no_speech`. Under five minutes is `too_short`, and its transcript is kept. Otherwise Claude fills `inputs/notulen_template.md` in the meeting's language, and the row becomes `committing`.

The prompt names a speaker only when the audio proves who it is; otherwise the minutes say "Speaker 1". The GPU worker is a separate image that imports nothing from the dashboard, so the prompt, the floors and the attendee cleaning are written twice. Tests keep the two copies in step.

### Committing and correcting

```
committing -> GitHub Contents API -> complete
<NOTULEN_TARGET_PATH>/<YYYY-MM>/<alice_bob>/<YYYY-MM-DD>.md
```

`complete` means the file is in `NOTULEN_TARGET_REPO`, committed with `GITHUB_NOTULEN_PAT`. A loop retries every 30 seconds with no time limit, and pages the assistant once an hour when minutes have waited more than 15 minutes.

A second meeting on the same day lands on `-2`; nothing is ever overwritten except by a person's correction. The editor saves over the committed file with the version it read, so two people editing at once collide instead of losing an edit. Deleting a recording removes the audio, the row and the file.

### Handing on to the assistant

- a recording that stopped sending audio for 90 seconds
- a GPU job Kubernetes failed
- a GPU dispatch that fell back to CPU
- minutes stuck uncommitted

Every page this dashboard raises goes to the assistant's `/webhook/notify`, never straight to a phone. The "Send to the assistant" button posts the corrected minutes to the assistant's `/webhook/notulen`, which opens or updates the issues for the action items and returns their numbers. The button fires once: the claim is written before the call, and given back only if the request never arrived.

### Who is in the room

The page knows the signed-in person (`Remote-User`, else `Remote-Email`) and the `NOTULEN_ATTENDEES` list, offered as chips. Anyone else is typed in. The same name in two spellings counts as one person, because the number of attendees is what the GPU worker uses to fold stray voices back into real ones.

## apps/documents

### What it makes

| Input | Output |
|---|---|
| a YAML data file | an A4 PDF in the house style |
| `design/tokens.css` | colors, fonts and spacing, injected as CSS variables |
| `design/company.yaml` | the issuing company block: name, address, IBAN, registration and VAT numbers |

`python -m documents <type> --data file.yaml --out file.pdf` renders `templates/<type>/template.html` with Jinja2 and turns it into a PDF with WeasyPrint. `--sample` renders the type's bundled `sample-data.yaml`, which uses only placeholder names. The invoice is the reference type; a new type is a copy of its folder, adapted.

Template CSS uses only design variables (`var(--color-primary)`, `var(--font-body)`), so every document follows a restyle of `design/`. Field labels have English defaults; an optional `labels:` map in the data file translates them, so one template serves any language. Data values are HTML-escaped. An empty company profile renders cleanly, without dangling "IBAN" or "Reg. no." lines.

### What it needs

WeasyPrint loads pango, pangoft2, harfbuzz and fontconfig at import time. When they are missing, the error names the apt packages to install. One test renders a real PDF and reads it back with pypdf; it skips with that same install line when the libraries are absent.

### Sending

`infra/email/send_document.py` sends one email, optionally with the PDF attached, through Scaleway's Transactional Email API, because Scaleway blocks outgoing SMTP. It reads `TEM_PROJECT_ID`, `TEM_FROM` and `TEM_REGION` from the environment or the harness env file, and the secret key from `SCW_SECRET_KEY` or the `scw` CLI. It has no default project. Nothing is sent without the owner seeing the document and saying yes.

## apps/website

### What it serves

```
browser -> Caddy (HTTPS, bare domain) -> website:80 (nginx) -> apps/website/ + /design/
```

The public site is static files in `apps/website/`, served by a stock `nginx:alpine` image.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=website) The Dockerfile sits beside the page and builds from the repository root, so it can copy both the page and `design/`. The page links `/design/index.css` and nothing else from the design system, so a change to `design/tokens.css` restyles the website, the documents and the dashboards together. The company profile and the build files are removed from the image, so they are never served.

### How it changes

Edit `apps/website/index.html`, or add pages and images beside it, and push to `main`. The deploy cycle rebuilds the `website` service, defined in `apps/website/compose.yml` and included by the root `docker-compose.yml`, within about two minutes. A healthcheck fetches `/` every 30 seconds. In the cluster the same image runs as two replicas in namespace `website`, and the deploy loop sets it by digest.

### design/

| File | Holds |
|---|---|
| `tokens.css` | the values: `--color-*`, `--font-*`, type scale, spacing, radii, shadows, status colors |
| `base.css` | plain HTML defaults plus `.lead`, `.muted`, `.eyebrow` |
| `index.css` | the one entry point, importing the two above |
| `company.yaml` | the company profile, every field empty until onboarding fills it |

Brand colors appear in both `company.yaml` and `tokens.css`, and onboarding keeps them in sync. A custom font is a `.woff2` in `design/` plus an `@font-face` block in `tokens.css`; until then the system font stack is used.

## infra/platform

<!-- Budget: about 950 words; four subjects (the login, the vault, the cluster and its services, the shared library), each explained once. -->

### One login across subdomains

| Host | Goes to | Login |
|---|---|---|
| `DOMAIN` | website | public |
| `cloud.DOMAIN` | Nextcloud | Nextcloud's own, password and second factor |
| `auth.DOMAIN` | oauth2-proxy | sign-in and sign-out pages |
| `notulen.DOMAIN` | notulen | single sign-on |
| `deploy.DOMAIN/hooks/*` | webhook service on the host | HMAC signature, no login |

Caddy is the only container with public ports.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=caddy) It gets certificates from Let's Encrypt on its own once DNS for these names points at the server.

Nextcloud is where accounts live.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=sso) You create a person there, they set a password and enrol an authenticator app, and from then on that one account opens every private subdomain. Nextcloud's oidc app makes it an identity provider; oauth2-proxy is the client that logs in against it.

On every request to a private subdomain, Caddy asks oauth2-proxy whether the browser carries a valid login cookie. Yes means Caddy forwards the request with the person's account id in `Remote-User` and their email in `Remote-Email`. No means a redirect to Nextcloud and back. Caddy first deletes any `Remote-User` the browser sent itself, so an app can trust the header.

The cookie is set on `.DOMAIN`, so one login covers every subdomain. It lasts 8 hours and is never refreshed. The Nextcloud session behind it lasts 7 days, so when the cookie runs out the browser is signed straight back in, and Nextcloud checks on the way that the account still exists. A disabled account loses access within 8 hours.

Adding a private subdomain: a Caddyfile block that imports `oauth2_auth`, the name in `SSO_SUBDOMAINS`, and a run of `infra/nextcloud/oidc-setup.sh`. Two-factor login is enforced for everyone once `NEXTCLOUD_2FA_ENFORCE=1`; the script refuses while any account has no second factor, because that account would be locked out.

### The vault

```
laptop: config/harness.env + secrets   (a private file)
  -> infra/secrets/push-env.sh          (scw CLI, new version)
  -> Secret Manager secret harness-env
  -> infra/secrets/pull-env.sh          (server, every deploy cycle)
  -> $HARNESS_ROOT/harness.env          (mode 600)
  -> docker compose --env-file
```

All configuration is in three places. `.claude/project.json` names the GitHub repository and the people. `config/harness.env` holds settings that are not secret, such as the domain and the region. The vault is one Scaleway Secret Manager secret holding the whole environment: those settings again, plus every password and token listed in `config/vault.env.example`.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=vault)

Each change is a new version of that secret. The server pulls the latest version every deploy cycle, so a pushed change is live a few minutes later without logging in to the server. If the pull fails, the server keeps the file it already has and the stack carries on.

The Scaleway keys themselves are not in the vault. The laptop uses the key from `scw init`. The server has its own key, allowed to read this one secret, in `$HARNESS_ROOT/scw-secrets.env`, which onboarding writes once and git never sees.

### The cluster

```
config/harness.env   (CLUSTER_*, POOL_*)
  -> infra/cluster/create-cluster.sh    (plan; --apply creates or adopts)
  -> one cluster in SCW_REGION, CLUSTER_ID written back
  -> pools per pool role: preferred, then stock fallbacks
  -> infra/cluster/apply.sh             (renders infra/k8s/)
  -> namespaces, PriorityClass deploy-loop, ClusterSecretStore vault
  -> one namespace per service, each on the system pool role
```

The cluster is the base every later piece runs on. Its pools are grouped by pool role: `system` for everything without a role, `gate` for the deploy gate's test runs, `media` for calls, `gpu` for transcription. A pod asks for a role through the node label `POOL_ROLE_LABEL`, never for a pool by name.

Each role has a preferred pool and stock fallbacks: empty pools of other machine types and other zones of the region, carrying the same label, taint and root disk. When the provider has no stock of the preferred type, the autoscaler's priority expander tries the fallbacks in order: other types in the home zone first, because a pod with a zonal volume can only follow there, then the other zones. A role given one zone never leaves it.

Pods get secrets through the ClusterSecretStore `vault` over the same Secret Manager secret the server pulls. An ExternalSecret writes the whole vault into Secret `harness-env` in each namespace that may read it, and a second splits it into one key per setting, `harness-keys`, so each container names exactly the keys its compose service was given and no key is ever written into a manifest.

Every long-running service of the compose stack is declared here too, one namespace each, so an exec grant on one never reaches another. Caddy sits behind one cloud load balancer, with the compose Caddyfile's upstreams rewritten to cluster Services at every apply, so it cannot go stale. A nightly job snapshots every volume labelled as holding state. Compose stays the live stack until the deploy loop runs here.

### The shared library

| Piece | What a dashboard gets |
|---|---|
| `make_dashboard_app` | a FastAPI app with an activity log, AI-call costs, health checks and a sync heartbeat already wired |
| `get_user(headers)` | the signed-in account from `Remote-User`, or `anonymous` |
| `get_db()` | a connection to the stack's Postgres, or None when it is down |
| `call_claude` | the one way to call Claude; every call is logged with tokens and cost |
| `/static/dashkit/*` | shared JavaScript and `base.css` |

dashkit lives in `lib/dashkit` and every dashboard is built on it. A dashboard keeps working when the database is unavailable: the activity log comes back empty and health checks answer 503, but nothing crashes. Activity and cost logging never break the feature being logged. Table names reach SQL only after a check that they are plain lowercase identifiers. `base.css` takes its colours and fonts from `design/tokens.css`, so restyling every dashboard is an edit to the tokens file.

## infra/deploy

```
push to main -> Caddy -> deploy-webhook
deploy-webhook -> commit-gate Job: a verdict for every queued commit -> ConfigMap commit-gate-verdicts
deploy-webhook -> deploy-cycle Job: lock -> fresh -> verdicts -> pick one commit -> export -> gate -> build -> roll out -> apply -> bookmark
```

### The loop

Pushing to `main` is the deploy.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=deploy) In namespace `deploy`, CronJob `deploy-cycle` runs `infra/auto-deploy/auto-deploy.sh` every two minutes, and the push receiver `deploy-webhook` (behind Caddy at `deploy.<DOMAIN>/hooks/*`) checks the push's HMAC signature and starts a cycle and the lane at once. A cycle takes the lock, fast-forwards its clone on volume `deploy-state`, and picks ONE commit past the bookmark, ConfigMap `deploy-bookmark`.

That commit is the trusted head when the lane has one (below), else the newest commit green for its exact tree, else the next one. A red on a range nobody judged would name nobody's commit.

The commit is exported with `git archive`. The gate runs as one gate Job per module group, all at once on the gate pool, each a full clone in the gate image the commit's `infra/gate-image/pin` names by digest. The stack's own images are built by BuildKit (Deployment `buildkitd`) and rolled out by digest; a rollout that does not finish is undone. The bookmark moves only when everything shipped.

When a shipped tree's `infra/deploy_loop` differs from the `deploy-cycle` image in use, the cycle builds it, runs `image/smoke.sh` in a pod, and tries it on CronJob `deploy-cycle` alone. CronJob `deploy-cycle-image-guard`, still on the old image, moves every other object onto it after one green cycle, or puts the cycle back after two failed ones or 30 minutes without a green one.

### The gate and its three verdicts

| Verdict | What it means | What happens |
|---|---|---|
| green | pytest exited 0 | cached for this commit, scope and tree; ships |
| red | the run happened and said no | nothing ships; a failing test run is cached |
| amber | the run could not start or finish | nothing ships; never cached, never reverts anything, retried next tick, alerts after three |

Red covers a failing test, code that would not load, a run that collected nothing, a test with no register row, and a test that passes with its own break applied. Amber covers a gate Job that could not start, a failed `uv sync`, a timeout and a run killed or out of memory. A gate that could not run must never look like one that passed.

`tools/gate_scope` follows the import graph and the register's triggers to pick the tests a change can reach; anything it cannot reason about runs the full suite. Narrowing is licensed only while the nightly unselected run (`tools/gate_scope/audit_full_run.sh`) is green; with none in 48 hours, or one proven miss, every commit runs the full suite.

### The verifying lane

| Lane verdict | Its reason field | What the cycle does |
|---|---|---|
| green, break replay passed | `replayed` | counts towards the trusted head |
| green, covered by a batch | `batch:<sha12>` | counts, never as the head itself |
| green, no replay | `none` | gates that commit itself |
| red, the commit's | `regression` and the like | nothing; the lane's eject acted |

CronJob `commit-gate` (`infra/auto-deploy/commit_gate/cluster_lane.sh`) judges every commit between the bookmark and `main` on its own tree, from its own clone on volume `commit-gate-work`. Its suites run as gate Jobs. It publishes its verdicts as ConfigMap `commit-gate-verdicts`, which step 18 copies into the cycle's state without replacing anything there.

A burst is one batch: verified once on its newest commit, halved only while red, until one commit is red on its own run. That red is sorted like the cycle's, and when it is the commit's the lane's own eject takes it back.

The trusted head is the newest commit up to which every commit has a replayed green, a batch green, no reachable test, or a red a later covered commit reverts (`python3 -m infra.deploy_loop trusted-head`). The cycle ships it without running pytest again. Without a narrowing licence only a `full` green counts.

### The register

Every test has one row in `tools/test_register/_rows.py`: the module it pins, the paths that select it, and the break, an exact edit that must turn the test red. `python3 -m tools.test_register --new <module> --says "<invariant>"` writes a test and its row together. The gate refuses a tree whose tests and rows disagree, and after a green suite replays the break of every row the commit touched: a test that still passes is red.

### The eject

The test files a red names run again in a fresh gate Job and on the last good tree. A test that passes the second time, or fails there too, is quarantined with an issue for its owner, and the suite runs again without it. Only the commit's red stays red.

The eject, armed by the shipped config (`EJECT_ENABLED=1`), reverts that one commit, only when its parent is known good, and never its own commits. The lane and the cycle each have an arm; the second sees the first one's `Auto-Eject:` revert on `main` and stands down. The work goes back on a branch `auto-eject/<sha>` with a draft pull request. Never press merge on it: fix it there and land it as one squashed commit.

## infra/observability

### Alert path

```
Prometheus (rules, every 30 s)
  -> Alertmanager (groups, repeats: critical 1 h, warning 4 h)
    -> alert-relay
      -> assistant /webhook/alertmanager -> triage -> Signal
      -> or, while the assistant-down flag is fresh: signal-cli /v2/send -> SIGNAL_RECIPIENT
```

Prometheus scrapes itself, Alertmanager, node-exporter and the blackbox exporter, and probes every subdomain in `PROBE_SUBDOMAINS` over HTTPS. When a rule fires, Alertmanager posts the group to the alert-relay, which hands it unchanged to the assistant.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=observability) The assistant decides what reaches the phone and how it is worded. Severity changes how often a still-firing group is repeated; it never decides whether it is delivered.

### Rules

| Alert | Fires when | Severity |
|---|---|---|
| TargetDown | a scrape target is down for 5 minutes | critical |
| EndpointDown | an HTTPS probe fails for 5 minutes | critical |
| DiskAlmostFull | a filesystem is over 85 percent for 10 minutes | warning |
| MemoryPressure | under 10 percent of memory is available for 10 minutes | warning |
| ContainerRestarting | a container restarts more than twice in 15 minutes | warning |
| UnitFailed | a `harness-*` systemd unit is failed | warning |
| TextfileCollectorStale | the collector has not run for 10 minutes | warning |
| DeployLoopStale | no successful deploy cycle for 6 hours | warning |
| CertificateExpiring | a certificate expires within 14 days (3 days: critical) | warning |

Every rule writes a summary and a `first_action` naming the command to run. The deploy loop proves it is alive by writing `harness_deploy_last_success_timestamp_seconds` into the textfile directory after each good cycle.

### Deadmen

| Job | Every | Watches | Pages when |
|---|---|---|---|
| `harness-assistant-deadman` | 60 s | the relay's `/assistant-check` | 3 failures in a row |
| `harness-gateway-deadman` | 60 s | the assistant's Signal gateway health route | 3 failures in a row |
| `harness-dead-mans-switch` | 15 min | the Watchdog alert in Alertmanager | unseen for an hour |
| `harness-signal-send-canary` | 30 min | the bot sending a message to itself | it tells you once sending works again |

A watcher cannot report through the thing it watches, so every deadman page goes straight to signal-cli, never through the assistant. Three misses, about three minutes, count as down; one miss is usually a restart. A down page goes out once and repeats at most hourly.

The assistant deadman asks the relay rather than the assistant, because the relay is the road an alert travels. Healthy means both: the assistant's `/health` answers, and the assistant accepts the relay's bearer on an empty alert envelope. An assistant can answer `/health` while refusing every delivery with a 401, and that has to count as down.

### Raw road

When the assistant deadman sees three failures it writes `/var/lib/harness/assistant-down`, rewritten every minute while the outage lasts. While that file is at most 300 seconds old, the relay stops forwarding and sends each alert group to Signal itself: one plain message per group, the same group at most once per half hour. An older flag means the deadman stopped writing, so the relay treats the assistant as up.

The Watchdog alert fires constantly and Alertmanager routes it nowhere. If Prometheus, a rule file or Alertmanager breaks, the Watchdog stops being refreshed, and the dead-man's switch pages from outside the pipeline.

### Unit failures

Every `harness-*` timer job carries `OnFailure=harness-notify-failure@%n.service`. A failure sends one Signal page with the unit name, the last line the failing run wrote, and its journal tail. The same failure is held for 12 hours; a different failure of the same unit pages at once. The Signal canary's own failure is not sent over Signal, and `harness-assistant*` failures stay quiet while the assistant-down flag is fresh.

### Where it runs

| Piece | Where |
|---|---|
| prometheus, alertmanager, node-exporter, blackbox-exporter, alert-relay | `infra/observability/compose.yml` |
| deadmen, canary, textfile collector, failure pager | systemd units, installed by `infra/observability/systemd/install.sh` |
| state, flag, textfile metrics | `/var/lib/harness` |
| all of it, in the cluster | namespace `monitoring` (`infra/k8s/monitoring`); the deadmen, the switch and the canary are containers in the alert-relay pod, on its claim `alert-state` |

Everything is published on loopback only: Prometheus 9090, Alertmanager 9094, relay 9095. Read Prometheus through an SSH tunnel.

The cluster Prometheus loads `rules/cluster.yml` instead of `harness.yml`: kube-state-metrics stands in for the textfile collector, so a restarting or memory-killed container, a failed Job in any namespace, a CronJob an hour past its run and a workload short of ready pods each page. A job whose pod ends with its run, the deploy cycle and the canary, pushes its gauge to the Pushgateway. The switch runs in the cluster it watches, so a whole-cluster outage silences it too.

## infra/gpu

### What the tier is

```
dashboard -> KUBECONFIG_B64 -> GPU cluster, namespace $GPU_NAMESPACE
record pressed -> placeholder Pod (holds a GPU)
recording done -> notulen Job (preempts it) -> worker writes row -> dashboard commits
```

The GPU tier is optional. Without it notulen transcribes on the server's CPU, which takes hours for a long meeting; with it a meeting takes minutes. It is a Scaleway Kapsule cluster with one or two GPU node pools that scale to zero, so a node bills only while it transcribes.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=gpu)

The dashboard talks to it with its own service account. The worker writes its result straight into the stack's Postgres, which therefore listens beyond the server (`POSTGRES_BIND_ADDRESS=0.0.0.0`), with the database password as the only guard.

### The pieces

| Piece | Where | What it does |
|---|---|---|
| Worker image | `infra/gpu/Dockerfile` | pyannote, whisper-large-v3, Claude; public pytorch base |
| Builder | `infra/gpu/build-worker.sh` | builds on a temporary instance in the region, pushes to the registry namespace `SCW_REGISTRY` |
| Stock finder | `infra/gpu/pick-gpu.sh` | first GPU type with stock, cheapest first, across zones |
| Cluster objects | `infra/gpu/apply.sh`, `k8s/` | namespace, service account and Role, PriorityClass, worker Secret, ConfigMap, pull Secret |
| Dashboard key | `infra/gpu/make-dashboard-kubeconfig.sh` | a kubeconfig with a static token, base64 into `KUBECONFIG_B64` |
| Job code | `infra/gpu-jobs/gpu_jobs` | builds and posts the Job and the placeholder |

### Turning it on

1. Create a cluster with one GPU pool named `$GPU_POOL`, autoscaling from zero, tainted `workload=notulen:NoSchedule`.
2. Run `build-worker.sh v1` and put the printed image in `NOTULEN_WORKER_IMAGE`.
3. Set `DB_URL_OPS` and the Postgres bind, then run `apply.sh`.
4. Put the kubeconfig in the vault and set `GPU_ENABLED=1`.

### Waiting for a GPU

A GPU node can take half an hour to start when its zone is short of cards. Pressing record therefore submits a placeholder Pod at the `notulen-gpu-warmup` priority (-5). The autoscaler starts a node during the meeting, and the real Job takes the placeholder's place by preemption when the recording ends. The placeholder ends itself after six hours.

The Job has no deadline of its own, so a meeting waits out a stock-out instead of being killed while still waiting. The worker's run is capped at four hours once a node accepts it. A second pool in another zone, listed in `GPU_POOL`, gives a meeting somewhere else to go.

### Credentials

No credential is ever written into a pod. The worker reads the database URL, the Anthropic key, the S3 keys and the optional Hugging Face token by reference to the `notulen-worker-secrets` Secret, which `apply.sh` builds from the vault copy at deploy time.

The dashboard's account may only create, list and delete Jobs and Pods in its own namespace. `MAX_DAILY_JOBS` (50) caps how many Jobs a day can start.

### What can go wrong

- A dispatch that fails falls back to CPU and pages once an hour.
- A Job Kubernetes marks failed is caught within a minute: its row is failed and the assistant is paged.
- A dashboard restart leaves running Jobs alone.
- A node that comes up without a usable GPU points at the provider's NVIDIA driver, not this code.

## infra/media

### What runs on the media node

```
media pool role, one node, host network
  media-stack pod     caddy, nats, janus, signaling, recording
  MEDIA_PUBLIC_IP     the node's flexible IP: DNS, Janus's ICE candidates
system pool role
  media-ip-reattach   every minute: the IP onto the one Ready media node
```

| Service | Job |
|---|---|
| `signaling` | Nextcloud Talk's High Performance Backend, pinned image |
| `janus` | the media server every call's audio and video passes through |
| `nats` | the message bus between the two |
| `recording` | the Talk recording server, a headless browser in the call |
| `caddy` | HTTPS for `signaling.DOMAIN` and `recording.DOMAIN` |

The calls run on the one node of the `media` pool role, so no other node opens the UDP range 20000 to 40000 that media needs.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=media) The five services share one pod on the node's own network, so they reach each other on loopback, and only Caddy's ports and the UDP range listen outside it.

Janus and the recorder are images built from upstream source at pinned commits, because the recorder touches every meeting's audio and video. There is no TURN relay, so a network that blocks direct UDP cannot join a call.

Janus tells every browser to send media to `MEDIA_PUBLIC_IP`, so that address must sit on the node, and a replaced node comes back with a new address. A CronJob on the system role checks every minute which Ready node carries the media role and moves the IP onto its server when it is not already there; with none or two it refuses to guess. Until the IP is on the node, the pod waits before Janus starts.

The media area is optional: `infra/cluster/apply.sh` skips it while `MEDIA_PUBLIC_IP` is empty, so no media node bills while calls are off. Caddy's certificates and recordings Talk has not fetched yet sit on claims the nightly backup snapshots.

### How a call becomes minutes

```
assistant: create_video_call -> a Talk room, the recording bot enabled on it
call starts -> Talk webhook -> notulen /api/talk/webhook -> start recording
call ends   -> Talk webhook -> notulen fetches the file -> S3 -> minutes job
```

Nextcloud Talk runs inside Nextcloud; the media node only carries its calls and records them.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html#focus=talk) The recording bot, installed by `infra/media/connect-talk.sh`, posts every call event to notulen, signed with `TALK_BOT_SECRET`. On `call_started` notulen asks Talk to record, retrying while Talk answers that the call is not recordable yet. On `call_ended` it finds the newest recording in the room owner's `Talk/Recording/<token>/` folder, refuses one it already holds or one older than the call, and hands it to the same pipeline as a recording made in the browser. A call over two minutes with no recording pages the assistant.

The assistant's account owns every call room, which makes it the moderator allowed to start a recording and the account the file lands with. Only rooms the assistant creates are recorded, and the room is told when recording starts; telling people beforehand is the organiser's job.

## workflow/loop

One task is one conversation, from "I want this" to production without leaving the chat: grill,
to-prd, code, close.

### Where a turn starts

| Way in | What it is |
|---|---|
| A new idea | something not there yet |
| A person using it | what they found by hand |
| A page from the assistant | the watch, no person |

No ticket exists first: the issue is written later in the turn.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=spark) All
three arrive as one messy prompt: pasted conversations, screenshots, links.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=dump) A page of thinking
rather than a task goes to `/brain-dump`, which sorts it into claims, questions, decisions and
vision. An effort too big and too foggy for one session goes to `/wayfinder`, which charts it as
a PRD whose route is unknown.

### Nothing is built before this

| What is read | Where |
|---|---|
| The glossary | `CONTEXT.md` |
| The bible chapter | `docs/bible.md` |
| The area overlays | `<area>/CLAUDE.md` |
| The decisions | `docs/adr/` |
| Prior issues | GitHub, open and closed |

The `grill` skill forces those reads before the assistant answers, and what comes back is one
question, not a summary; a session remembers nothing of the last one.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=orient) **Facts are looked up, never
asked.** What reaches the person is judgment, taste, priority and risk.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=decision)

**Whatever settles is written down at once**: a term in the glossary, a trade-off on its issue or
as a proposed ADR, an invariant in its chapter.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=capture) The plan leaves the chat as a GitHub issue written
like a PRD (`to-prd`); conversations are throwaway, the issue outlives them, and the board over it
is where work is tracked.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=board)

### Write it, cut the old, prove it

```
one commit, straight on main
  the code
  the module's drawing
  the bible text claiming each box
  minus what it replaces
  plus the test that pins it, with its register row
```

No branch and no pull request keeps a task in one conversation and leaves no review queue
(ADR-0001).[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=build) A session that writes files works in its own worktree, set up with
`bash .claude/scripts/worktree-setup.sh`, and pushes with `git push origin HEAD:main`. The drawing
and the chapter are written in the same pass as the code, through the `diagram` MCP and the
`case-study` skill, and linked into the issue only when a human is needed.

**Dead code is a defect, not an archive.**[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=deleted) A module exists only while deleting it would scatter
real complexity across its callers.

### What stops a bad commit

| Check | Where | When |
|---|---|---|
| The torn-commit guard | `.claude/hooks/pre-commit` | every commit |
| The test register and break replay | `.claude/hooks/local/pre-commit` | every commit |
| The `audit-*` skills | a session | before a push that matters |
| The deploy gate | the server | every commit on `main` |

A test and the change that makes it pass go in one commit: the gate runs every commit against its
own tree, so a test that arrives one commit before its fix is red for ever.
A red sends the turn back to building.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=verify)

### After the push

| What runs | Cadence | What stops without it |
|---|---|---|
| auto-deploy on the server | a short timer, plus a push webhook | nothing ships |
| The eject, `EJECT_ENABLED=1` | on a red gate | bad commits stay on main |
| A person on the page | per change | nobody says it is right |
| The assistant on Signal[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=watch) | around the clock | production runs unwatched |

Pushing to main is the whole deploy interface.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=live) GitHub Actions never build or test; the server
posts its own commit statuses, so the commit page shows what was verified, built and made live.
The tests answer whether it broke; the person answers whether it is right, reading the drawing
against the code.[◈](https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack/loop.html#focus=check) What they find becomes the next messy prompt.

### Tuning Claude

- **Friction**: Claude gets something wrong for this team.
- **Rule**: the correction goes into `.claude/soul.md` or the root `CLAUDE.md`, with a name and a date.
- **Inject**: a `UserPromptSubmit` hook replays the soul rules into every prompt.
- **Behaviour**: the tuned default for everyone here, not just the person who corrected it.

Four mechanisms carry rules, one job each: **skills** (`.claude/skills/`, methodology),
**overlays** (`<area>/CLAUDE.md`, loaded with a directory), **hooks** (`.claude/settings.json` and
`.claude/hooks/local/hooks.json`, firing on a harness event), and **GitHub issues** (the durable
backlog). "From now on, when X" is a hook. The shared files under `.claude/` come from the starter
template and stay identical to it; this repository's own behaviour goes in `.claude/hooks/local/`
and its own skill directories.

### The board, commits and memory

- **Board**: one GitHub project, from `setup-github.py --board`. Status is a column (`LowPriority`, `Not started`, `In Progress`, `Review`, `Done`), set with `.claude/scripts/board-status.py`; triage state is a label.
- **Commits**: imperative and scoped, `module: description`. A subject that needs "and" is two commits.
- **Memory**: linked into `.claude/memory` by `bootstrap.sh` and committed by the `Stop` hook, so it survives a wiped machine. Nothing secret goes in it.

## workflow/module-standard

The house standard for a module, so a reader can predict the shape of code they have not seen
yet. Each rule below is a line an auditor can mark pass or fail.

### Tiers and layout

```
<package>/                  # Tier S: one deep module, up to about 8 files
  __init__.py               # the seam: docstring module map, re-exports, __all__
  <one_concern>.py          # one public callable, helpers _prefixed
  tests/test_<invariant>.py # named for what it pins

<package>/                  # Tier M: a 5-layer deep module
  entrypoints/              # CLI mains and worker loops; config assembled here only
  gates/                    # validation as data: (ok, reasons)
  core/                     # the domain logic
  shared/                   # helpers used by 2 or more layers
  inputs/                   # reference data, declared committed or ignored
```

The tier is declared in the package overlay or docstring. A directory with more than 12 entries is
missing a boundary. No loose files outside the carve-out (ADR-0002). No `sys.path` tricks in
source; a container's import layout is solved in its Dockerfile.

### The interface

- **`__init__.py` is the declared seam**: a docstring module map, curated re-exports, `__all__`. An empty one invites file-path imports.
- **Every implementation file opens with its contract**: what it owns, what it does NOT own and where that lives, and a FAILURE POLICY for anything touching a database, the network or a subprocess.
- **One public callable per file.** Dataclasses, protocols, exceptions and enums in its signature ride with it.
- **Callers import the package**, never a file inside it and never a `_`-prefixed name.

### The implementation

- **One function, one purpose.** A docstring that needs "and" between two concerns is two functions.
- **No single-letter names** outside comprehensions and lambdas.
- **Heavy dependencies import lazily** inside the function that needs them, so the package and its tests import without them.
- **Env vars are read at call time**, not import time, and the config tree is assembled only at the entrypoint; `core/` takes primitives.
- **No DI container.** Collaborators are keyword-only parameters with production defaults; a port is introduced when a second adapter exists.
- **One home per value.** A constant declared twice will drift.

### Tests

```
python3 -m tools.test_register --new <module> --says "<invariant>"
python3 -m tools.test_register --replay <test file>
```

The first writes the test and its register row together; the second applies the row's break and
demands red. Every test has one row, and the row carries an exact edit that must turn the test
red, so a test that cannot fail is caught. A test checks against a synthetic ground truth: the
input it builds itself and an answer worked out without the code under test, never read from the
live machine, the clock, the network or production data. It crosses only the public seam, and its
collaborators are injected rather than patched.

A new test is sealed at birth. The commit hook refuses one whose source reads the live machine,
the clock or the network, then runs it in the gate image (`infra/gate-image/pin`, by digest)
with the network off and the clock 10,000 days behind.

| Level | Its tests |
|---|---|
| a module | one file, one test function |
| a module group | one file, one test function, of how its modules fit |
| the whole tree | the full suite |

A module has one clear purpose, so it has one test function. A second check is a parametrized
case of it, never a second function, and a module group is held to the same rule. `--check`
refuses both, past a debt list of the tests already over the rule, which only shrinks.

A red that is the test's fault never holds the deploy. When the failing test also fails on the
last good tree, or passes on a re-run of the same commit, the gate quarantines it (a line on its
row only the gate writes, an issue for its owner) and goes on without it, with no time limit.
A red that is the commit's is reverted and handed back at once.

### Failure tiers

| Tier | What | Rule |
|---|---|---|
| 1, structural state | database writes, state transitions, money, auth | propagate, fail loud |
| 2, best-effort side effects | alerts, notifications, cleanup | swallow and log, with the reason written at the site |

Fail-open or fail-closed is a named decision per seam, never an accident. A bare `except: pass`
is a finding in either tier. At `gates/`, failure is a return value, not an exception.

### The auditor's checklist

1. Tier declared and matching the tree; no directory over 12 entries.
2. `__init__.py` carries a module map and re-exports, or a written reason it does not.
3. One public callable per file; helpers `_`-prefixed.
4. Every module docstring says what it owns, what it does not, and its FAILURE POLICY.
5. Nothing outside the package imports a `_` name or a file path behind the seam.
6. One purpose per function; no single-letter names.
7. Heavy dependencies lazy; env read at call time; config assembled at the entrypoint only.
8. Every test has a register row whose break turns it red.
9. Tier-1 writes propagate; every Tier-2 swallow has its reason at the site.
10. Overlay, bible and docstring agree with the code; a sanctioned exception is written in both the overlay and the chapter.

## setup/reference
<!-- Budget: the operator reference moved out of the README (2026-11-01); a reader looks things up here rather than reading it through. -->

### Requirements and costs

The harness alone needs Linux or macOS (Windows through WSL2), `git`, `curl`, `python3` 3.12 or
later, [`uv`](https://docs.astral.sh/uv/), the GitHub CLI, Claude Code on a Claude subscription,
and a GitHub account, since the backlog lives in GitHub issues and a GitHub Projects board.

The stack also needs a Scaleway account paid by a normal bank card (prepaid and most virtual cards
are refused), a domain whose DNS you control, and an Anthropic API key for the assistant and the
minutes writer, billed apart from the subscription. The assistant needs a Signal number. `node` is
needed only to draw diagrams.

Rough monthly cost, as a shape and not a quote; check Scaleway's current prices.

| Item | When | Roughly |
|---|---|---|
| One server (DEV1-M class) and its IP | always | 15 to 25 euro a month |
| Secret Manager, one secret | always | cents |
| Object storage for meeting audio | always, grows | usage-based |
| GPU node | only while transcribing | depends on the card the picker finds |
| Media node (PRO2-XS class) and its flexible IP | only with video calls | one node, all month |
| Anthropic API | per message and per meeting | usage-based |
| Domain | yearly | about 10 euro |

Deleting the server does not stop the bill. The bucket, the secret, a registry namespace, a GPU
cluster and the media node keep billing until they are deleted too, so set a budget alert in the
Scaleway console.

### Configuration

Nothing is read from anywhere but these three places.

| Where | Holds | Committed |
|---|---|---|
| `.claude/project.json` | repository, owner, board, people, areas | yes, filled by `setup-github.py` |
| `config/harness.env` | settings that are not secret: domain, region, names, switches | no; copy from `config/harness.env.example` |
| the vault | the full environment including every secret, one Scaleway Secret Manager secret | no; keys listed in `config/vault.env.example` |

`infra/secrets/push-env.sh <file>` pushes the vault. The server pulls it into
`$HARNESS_ROOT/harness.env` each deploy cycle, so a change is live a few minutes later.

### Permissions and blast radius

The workflow assumes an unattended agent, which means `claude --dangerously-skip-permissions`. That
hands the agent your shell with your rights. During setup it holds an account-wide Scaleway key and
can create and delete servers, buckets and clusters and run up a bill. It holds GitHub
credentials, and a push to `main` is a deploy.

The hooks block a handful of shapes (force pushes, `git reset --hard`, new markdown files, torn
commits). They are not a sandbox, and nothing stops `rm -rf` or a Scaleway delete. With the flag
on, you are the review step. The alternatives are running without the flag and answering the
prompts, or running the agent in a container or virtual machine you are willing to lose. WSL2 is
not a boundary, since your Windows drives are writable from inside it.

### One harness across several repositories

The shared harness is `.claude/settings.json`, `.claude/soul.md`, the skills, the index and the
hooks beside the settings. Those files stay as they came. A repository's own hooks go in
`.claude/hooks/local/`, listed in `local/hooks.json` and run by `.claude/hooks/dispatch.py`. A
change to a shared file is made once, in the repository treated as the source, and copied to the
others, while each keeps its local hooks, its `project.json`, its glossary and its bible.

For a copy of this repository, the source is this repository. `/update` fetches it as the
`upstream` remote and merges, keeping your website, design, business folder, memory, glossary and
any file you created, and taking upstream's machinery. The authors keep their own repositories in
step with a lock file that pins each shared file by hash and refuses hand edits; that script is
not in this repository.

### Without a Claude subscription

The server keeps running, since the stack, the deploy loop, monitoring and the gate never call
Claude Code. The assistant and the minutes writer keep working on the separately billed Anthropic
API key.

The loop, the skills, the hooks, setup, `/update`, `/troubleshoot` and adding a page all stop.
They exist only as prose an agent reads, and no script replaces them.

### Components

<a href="https://babon-innovations-b-v.github.io/babon-harness/diagrams/stack.html">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/images/stack-dark.png">
    <img alt="The stack: your system, Claude Code with the harness, reaches one server through Caddy in a browser and talks with the assistant over Signal both ways, so its questions, pages and briefs come back; behind it run Nextcloud with Talk, notulen, the assistant and Prometheus; the assistant books meetings on the calendar and makes Talk call links; calls run through the media box and a recorded call becomes minutes committed to GitHub; a push from your system reaches the deploy loop; object storage, the vault and an optional GPU tier are rented beside it" src="docs/diagrams/images/stack-light.png">
  </picture>
</a>

*The stack. The picture opens the interactive page; its Your system box opens the loop.*

#### Platform

Caddy is the only public container. Accounts live in Nextcloud, and oauth2-proxy logs each private
subdomain in through it, so one login with a second factor opens everything. The dashboards are
built on the shared library `lib/dashkit`. `onboard` does the setup below; by hand it is:

1. Fill in `config/harness.env`, then assemble the vault file following `config/vault.env.example`.
2. Push it with `infra/secrets/push-env.sh`.
3. Point DNS for the domain and its `cloud`, `auth`, `notulen` and `deploy` subdomains at the server.
4. Run `docker compose --env-file $HARNESS_ROOT/harness.env up -d`.
5. Run `infra/nextcloud/oidc-setup.sh` on the server, put the two keys it writes into the vault,
   and push again.
6. Enrol a second factor for each account, then set `NEXTCLOUD_2FA_ENFORCE=1`.

#### Deploy

CronJob `deploy-cycle` in the cluster, every two minutes and on every push, runs
`infra/auto-deploy/auto-deploy.sh`. It takes one commit per cycle, tests it in its own tree, rebuilds only the services that changed and
checks they boot. A red or amber verdict ships nothing. The commit's GitHub page shows `vm/gate`,
`vm/build`, `vm/boot` and `vm/deploy`.

- First-time setup: the steps in `infra/k8s/deploy/CLAUDE.md`, ending with the push webhook at
  `https://deploy.$DOMAIN/hooks/deploy`.
- `DEPLOY_STATUS_TOKEN` needs commit statuses and deployments write.
- A test needs a row in the register: `python3 -m tools.test_register --new <module> --says "<invariant>"`.
- The eject is armed in the shipped config (`EJECT_ENABLED=1`) and needs contents and
  pull-request write on `DEPLOY_STATUS_TOKEN`; `EJECT_ENABLED=0` with `EJECT_DRY_RUN=1` tries it
  first. A quarantine is committed as `GATE_AUTHOR_EMAIL`; with that empty, no test leaves the
  gate for longer than one run.
- `AUTO_DEPLOY_ENABLED=0` pauses deploys; suspending CronJob `deploy-cycle` stops it.
- The gate narrows to the tests a change can reach only after the first green nightly audit, and
  the audit has no cluster runner yet, so every commit runs the full suite.

#### Observability

A page goes out when a target or endpoint is down, a disk is almost full, memory runs short, a
container keeps restarting, a `harness-*` unit fails, the deploy loop stops, a certificate nears
expiry, the assistant or its gateway is down, or the alerting pipeline itself goes quiet. Pages
reach `SIGNAL_RECIPIENT` through the assistant while it is up and straight over Signal while it is
down. Prometheus listens on loopback only: `ssh -L 9090:127.0.0.1:9090 <server>`.

#### Assistant

Upstream [OpenClaw](https://github.com/openclaw/openclaw), unmodified, holds the Signal
conversation. The tool door in `apps/assistant` triages alerts, decides what reaches a phone, and
hands the gateway its tools: calendars, the board, Prometheus, deploy status and the repository's
documents. It ships empty, with no people and no memory.

Over CalDAV against Nextcloud it reads the calendars in `AGENDA_CALENDARS`. Three switches, all
off until turned on, let it book a meeting on each attendee's calendar
(`ASSISTANT_AGENDA_WRITE=1`), put a Nextcloud Talk call link in a video meeting's location
(`ASSISTANT_VIDEO_CALL=1`), and send attendees that link on Signal shortly before the start
(`ASSISTANT_MEETING_REMINDERS=1`).

It needs `SIGNAL_*`, `ANTHROPIC_API_KEY`, `OPENCLAW_WEBHOOK_SECRET`, `OPENCLAW_GATEWAY_TOKEN` and
`AGENDA_*` in the vault. Link the bot number with signal-cli in native mode once, then run
`docker compose up -d` and `apps/assistant/workspace/apply.sh`; the deploy loop reruns `apply.sh`
after each deploy.

#### Meeting minutes (notulen)

A recording made on the phone at `notulen.$DOMAIN` becomes minutes in `NOTULEN_TARGET_REPO`,
under `NOTULEN_TARGET_PATH` (default `business/minutes`). Transcription runs on the CPU by
default, where an hour of audio takes hours. The optional GPU tier takes four steps in
[`infra/gpu/CLAUDE.md`](infra/gpu/CLAUDE.md); it costs GPU node hours and opens Postgres to the
internet with only the database password in the way.

#### Video calls and recording (optional)

Calls run on Nextcloud Talk, carried by an optional media node: the one node of the cluster's
`media` pool role, running [`infra/media/`](infra/media/CLAUDE.md)'s signaling server, the Janus
media server and the Talk recording server. Each call room the assistant makes has a recording bot in it. When the call
starts notulen asks Talk to record, and when it ends notulen fetches the recording from the room
owner's files and writes minutes, as for a browser recording. The room is told when recording
starts; telling people beforehand is up to the organiser.

`infra/cluster/apply.sh` applies it once `MEDIA_PUBLIC_IP` is set, and the recorder is built from
a pinned upstream commit. Turning it on is five steps in
[`infra/media/CLAUDE.md`](infra/media/CLAUDE.md), or phase 11 of `onboard`. There is no TURN
relay, so someone on a network that blocks direct UDP cannot join.

#### Documents, website and design

- `uv run python -m documents invoice --sample --out out.pdf` renders a sample invoice.
  WeasyPrint needs `libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libfontconfig1` on Debian or
  Ubuntu, `brew install pango` on macOS.
- Edit `apps/website/index.html` and push; the site follows `design/`.
- `design/tokens.css` styles the website, the documents and the dashboards together.
- Email needs a domain verified in Scaleway Transactional Email, plus `TEM_PROJECT_ID` and
  `TEM_FROM` in `config/harness.env`. Nothing is sent without an explicit yes.

#### Diagrams

The two pages above, `docs/diagrams/stack.html` and `docs/diagrams/stack/loop.html`, are drawn
from the specs in `docs/diagrams/specs/` with `python3 docs/diagrams/tools/build_free.py stack
stack/loop` and served by GitHub Pages from `docs/`. They were delivered with
`BIBLE_BASE_URL=https://github.com/Babon-Innovations-b-v/babon-harness/blob/main/docs`, so a box's
bible link opens the section on GitHub; set your own address before re-delivering them.

The README pictures in `docs/diagrams/images/` are compact layouts of the same pages
(`docs/diagrams/specs/readme/`), rendered dark and light by
`python3 docs/diagrams/tools/readme_images.py stack stack/loop`, which also needs the Python
`playwright` package.

`python3 docs/diagrams/tools/linkmap.py` checks the set. The `diagram` MCP server (in `.mcp.json`)
draws through `draw_module`, `link_module` and `check_diagrams`, with `node` for the archify
renderer. `BIBLE_BASE_URL` and `DIAGRAM_AUTHORS` are set in `config/harness.env`.

## physics/quantum-relativity

The hobby physics project in `physics/`: a simulation of the owner's **quantum-relativity theory**, which
tries to unify quantum physics and general relativity starting from a grid of cells rather than from
particles in space. Code that breaks a rule below is simulating something else.

Budget: this chapter holds the owner's theory rules in full, one place, changed often in conversation; splitting them across files would let them drift.

### The foundation

Owner, 2026-11-15, replacing every earlier picture of the structure:

1. **ABAB structure.** Two kinds of energy volume: **A**, the energy volumes that pull, and **B**, the joints, which push. Every energy volume is fully wrapped in joints: it never touches another energy volume, not on a face, an edge or a corner (owner, 2026-11-16). Joints may touch each other and form one connected web, like mortar between bricks; where joints meet are the seams. (Keeping both kinds fully apart is impossible in anything that fills space: wherever three or more cells meet along an edge, two of one kind meet there.) The sim starts from an actual grid of energy volumes, each wrapped in joints; the unevenness comes from slightly different energies, and one correction lets sizes follow energy, after which the rules bend, squeeze and stretch the grid.
2. **A pulls** on nearby A's, the pull spreading out and fading with distance. **B pushes** back, harder the more it is squeezed, steeply, and collapses past its limit.
3. **Energy sets size** for both kinds: more energy makes a volume smaller and more packed. A B is squeezed when the pull across it beats its own push (2026-11-17: its size is always what its energy sets).
4. **Everything else** (the leak, seams, drift, light, time) is how energy moves through this structure. Energy leaks from energy volume to energy volume only through joints (A to B to A); the more squeezed the joint, the less it lets through, and past the seal point nothing passes, so packed lumps seal themselves. Volumes deform: their size, shape and centre follow their energy and their neighbours' pull and push. They never change neighbours, except through joint collapse or volume split. A moving mass is a pattern of energy travelling through the grid, not a volume travelling. So energy flow has momentum (owner, 2026-11-15): energy flowing through a joint keeps flowing the same way unless a pull changes it, which lets a lump moving sideways past a bigger mass keep going while it falls, and orbit. Energy is a quantum fluid (owner, 2026-11-16): every Planck box holds whole quanta and a phase: the quanta are the energy density and the phase sets the flow (Gross-Pitaevskii style; layer 1's fluid has no push on itself, an open question). Inside a cell it flows without friction, and it can whirl only in whole steps; friction arises only in passing through a joint, as the joint's resistance (more squeezed, more friction; none passes past the seal point). Spin is an unbreakable whirl (owner, 2026-11-16): a whirl in the fluid that cannot unwind; its test is pending, and spin one half, two numbers per box, comes later. Friction destroys no energy: it turns directed flow into undirected jiggling, which is heat. Time is local (owner): the steps are only the computer's counter, and each region's clock runs at a rate set by its own energy concentration, joints included; the more packed, the slower everything there happens. Near a mass the pull squeezes the joints, so clocks there run slow even where there is little energy volume, as measured with GPS clocks. The drawing shows the simulation as it is and makes no corrections of its own.

**The engine, a fresh start** (owner, 2026-11-18). The engine before it had grown to 12,900 lines and basic gravity still did not work; it is in git history. The new one is built for scale from the start and in layers, each one something to look at: layer 1 one cell, layer 2 many cells and the first gravity test, layer 3 a billion cells on one graphics card. `physics/engine/` holds one directory per axiom, each with `definitions/` (what things are), `logic/` (what happens) and tests; no file is over 80 lines, and layer 1 is about 590 lines of code (730 with its blank lines). **Layer 1 is built:** one energy volume with its block of **Planck boxes** (whole quanta and a phase each). Two neighbouring boxes trade what their waves differ by, in whole quanta, so the total is exact. The whirl was rebuilt on 2026-11-19 (see where it stands below): a cell is three ring whirls, one for each direction, each a roll of boxes closed round an empty core. **Storage for a billion cells:** a calm cell is a rule of 5 bytes (its total and how its swirl is laid) and its boxes follow from the rule; only a cell that differs holds its own boxes (1.7 kB), and a step visits only those. A calm cell takes its own boxes exactly, and goes back with every quantum kept. A billion cells are 5 GB of rules. With 14.5 GB of a 16 GB card free, that leaves room for about 5 million cells with their own boxes when they lie together and about 2.5 million when they lie scattered, because each run of 512 cells that holds an own cell needs a 2 kB row in the map; whether gravity leaves that few is what layer 2 measures. The viewer (`physics/view`) shows the one cell as its swirl and nothing else: red streaks, each the way a tracer came that the fluid's own flow carries, solid where the fluid turns fast and clear where it stands, round the open core. No box is drawn. Measured (seed 1, 6 across, 3,000 quanta a box): after 1,000 steps the fluid still winds once on every ring with its core at 0.14 of the mean; the whole-quanta step is not reversible and its noise is not damped, so a cell left alone is further from the calm rule than counting noise after about 3,000 steps; with 20 quanta a box the swirl is gone in 100 steps. Chosen, not given by an axiom: the step size (`TURN`) and the 3,000 quanta a box. Taking a cell own and giving it back is done one cell at a time from the host in layer 1; doing it for many cells on the card is layer 2. The fluid steps plus quanta only. Not built yet: joints and the A B A B grid, the size law, crossing, the pull, limits, splits. Per axiom: the `CLAUDE.md` in each `physics/engine/axiom_*/`.

### The axioms

The owner's seven axioms (2026-11-16; an eighth, "nothing else", was dropped on 2026-11-17: with the seven done right it is a given). The engine is laid out by them: one directory per axiom, each with its definitions, its logic and its tests.

1. **Energy**: an always-moving quantum fluid, one whole signed number of quanta per Planck box at each moment of local time; the total is not fixed (owner, 2026-11-17): a squeezed joint makes more minus energy out of itself, and when a joint collapses its minus energy flips to plus and joins the volumes beside it (owner, 2026-11-19); zero is empty space, which does nothing.
2. **Cells are swirls**: plus only pulls, minus only pushes, both oscillate in whole steps. Energy is how fast a swirl turns: more energy, faster turn, smaller cell; plus and minus are the two directions of the turn (owner, 2026-11-19). Swirls turning the same way want to merge, which is the pull, but two turning the same way would grind where they touch; so a swirl turning the other way, a joint, lies between them and they mesh like gears. The swirl is not a pipe: its empty core is a ring closed inside the cell, so the swirl reaches the whole outside of the cell.
3. **Structure**: plus never touches plus; minus wraps every plus and forms one web. The cells are the grid (owner, 2026-11-17): a perfect A B A B pattern that never breaks and never flickers; a cell has no geometry of its own, only relations to its neighbours and a size set by its energy alone, more energy smaller, for volumes and joints alike. Lumps warp the grid by making their cells small. Planck boxes are not space: they sit inside each cell only to compute its fluid.
   A joint is a volume squeezed between the round energy volumes that pull across it: thinnest between two of them, fat where several meet, so the joints together read as a grid of crosses (owner, 2026-11-18).
4. **Crossing**: energy diffuses between plus swirls only through minus; that is the only friction, and what friction holds back stays as energy in the volume it came from (owner, 2026-11-18; there is no heat).
5. **Packing**: more packed is smaller, slower and pulls harder; pull spreads at a finite speed, weaker with distance; squeezed minus pushes back steeply.
6. **Limits**: a joint's energy rises only while it is squeezed, so a fresh joint is the fattest it will ever be and shrinks as it is squeezed; it collapses only past a limit, where the pull across it is so deep that nothing moving at the top speed could climb out (a black hole opening): its minus energy flips to plus and it merges with the pulling volumes beside it into one volume, which pulls harder, so the collapse can spread; a volume's leaking outside tears off as a layer of new volumes.
7. **Quantum**: each box's fluid also carries a phase, so waves add and cancel. Moving energy can split over several routes at once. Every split gets a direct pull between its halves, private to them, and its own oscillation, a shared state that keeps turning (not time, not a local clock), starting at the split at a speed set at that split that nobody can know. At each turn a pointer jumps to one of the halves' cells, with that cell's share of the energy as its chance. When something reacts with a half, or the energy loses its momentum, it settles whole in the cell the pointer is on.

How the grid comes to be (owner, 2026-11-19; this replaces tearing a volume in two, which grew a string): it is never laid ready-made; physics decides its shape. **The leak is the reason for the tear.** A volume is a whirl, and the whirl leaks: the bigger the volume (the lower its energy), the more it leaks. Where a volume has neighbours the leak crosses to them through the joints. Where it has none, the leaked outside tears off as a layer of new energy volumes all round it, like the layer of an onion, each new volume with a joint to the one it came from and to its neighbours in the layer; the joint is the place that tore, flipped to minus, and from then on it is squeezed by the volumes pulling across it. The new volumes leak too and shed a further layer outside them, and so on: the universe is a ball of cells growing layer by layer, its outermost cells always energy volumes, and this is its expansion. Cells must be seen to change size, and energy to keep moving between them. The leak is the swirl rubbing against the joint at its outer edge. Energy comes in whole quanta, so a leak can never be shared out evenly: which side an odd quantum goes to is decided by the swirl's own oscillation at that moment. That is where unevenness starts; the pull then makes it grow (owner, 2026-11-19). A volume's energy can never go to zero, and no extra rule is needed for that: the leak cascades. All the universe's energy starts in one volume, every volume leaks only a share of what it holds, and it is fed by the leak of the volumes further in (owner, 2026-11-19). Two volumes can never stand closer than their own widths: the joint lies between them and always has some thickness. A joint gives way only by collapsing, and it collapses only past a set limit of energy in the joint; that limit has to match the point where a black hole opens (owner, 2026-11-19; how to set it is open). The leak and the crossing are two separate things (owner, 2026-11-19): the leak simply happens, a property of the swirl; energy crossing to a neighbour in different amounts needs something to drive it: a force, inertia, momentum. Two neighbours holding different amounts do not even out by that difference alone. A black hole must be able to form later, from energy gathering, not only at the start (owner, 2026-11-19). Momentum means this (owner, 2026-11-19): energy carries its motion with it when it crosses, and meets resistance at each crossing of a joint.

Axiom 7 as the old engine read it (assumptions until the owner confirms; not built in the fresh version). **A split**: new outflow of a volume that goes into two or more volumes in one step has met a fork; it is one moving energy and its halves lie in the volumes that took it. The split's oscillation gets its speed at the split from its own draw. The pointer jumps at every whole turn by share. **A reaction** is a collapse or tear in a half's cell, or a scene's detector. **Losing its momentum**: at every crossing its halves make, the joint's friction takes direction off the split; when none is left it settles, every half drawn into the pointer's cell at once, in whole quanta. Open: a settling moves energy across the universe in one step, faster than light; no pull acts between the halves before settling; filter-angle tests on pairs of whirls.

### The rules

1. **Only relations are fundamental.** There is no background space: the grid of cells *is* space, a cell has only its neighbours and a size set by its energy, and distance, size and curvature are sums of cell sizes along the relations. Planck boxes are not space; they sit inside a cell only to compute its fluid (owner, 2026-11-17).
2. **Two kinds of cell.** **Energy volumes** are the cells that hold pulling energy. Every energy volume is wrapped on all sides by **joints**, pushing cells of their own; neither has a shape, only neighbours and a size. Joints touch each other and form one web; where two joints touch are the seams. The pull is a field spreading from cell to cell from the energy volumes' energy, fading with distance; joints carry it but add none. Which cells neighbour which changes only by joint collapse or volume split (owner, 2026-11-15); the pattern itself never breaks and never flickers (owner, 2026-11-17). The physics and every measurement read only energies, sizes and which cells neighbour which. In joint collapse the two volumes fall into each other; nothing is combined into a new shape.
3. **The grid is not uniform and not static**: its shape and size follow the energy in it. "Shape" is energy being more or less concentrated. More energy makes a cell smaller and denser, never bigger (owner, 2026-11-16).
4. **Energy is the only countable thing** (owner, 2026-11-16): it comes in whole quanta, the engine counts it as whole numbers so its books are exact, and every other quantity (size, pull, push, time, flow) is a property of how energy is concentrated. Each Planck box holds one whole number: plus means pulling energy (an energy volume), minus means pushing energy (a joint), the size is how much. Plus and minus never cancel. The total is not fixed (owner, 2026-11-17): a squeezed joint makes more minus energy out of itself, and a joint collapse takes the joint's energy off the rest (its own quanta go, and as many again off its two volumes), so black holes keep the total from running away; the merged, packed volume is where a black hole begins. **Energy density sets size**: more energy packed into a region makes it smaller and more packed, as length = constant ÷ energy (like E = hc/λ). A cell's size follows its own energy, so when a volume splits, each half holds less energy and grows: a split makes new room (owner, 2026-11-16), a candidate driver of expansion. No cell is held to one shape: the engine starts from blocks of boxes, but energy volumes and joints both bend, stretch and tilt by the rules.
5. **Joints oscillate**: joints are the springs of the grid; squeezed, they push back and overshoot, so they keep swinging smaller and bigger around their balance and are never perfectly still. Energy volumes do not vibrate on their own; their size follows their energy. A joint's swing changes its seams' width, so it changes the leak. Seams resist in grades, not just open or sealed: a wave travels straight through even resistance and bends or reflects where resistance changes, so squeezed seams near a lump slow it and bend it toward the lump, as gravity bends light.
6. **Drift is the leak**: every energy volume leaks energy through the joints around it, entering and leaving each joint only at its seams (where two joints touch, never through a joint's middle; the more squeezed the joints, the narrower the seam; past a seal point, reached before the push limit, the seam closes and nothing leaks, so a packed lump seals and holds steady, and only more squeeze collapses it) to its neighbours, carried by the energy's own flow and momentum, which pull steers toward packed regions (owner, 2026-11-16: momentum replaces the earlier fixed share-by-pull). Packed volumes gain and pack tighter, so lumps form; the volumes they draw from lose energy and grow, so empty regions expand, and space as a whole grows as lumps form. There is no separate drift setting.
7. **Attraction**: energy volumes pull on each other, mass attracting mass; a small, packed volume pulls harder than a large, thin one. Pull spreads through all of space, weaker with distance, the way gravity does (owner, 2026-11-16).
8. **Joint push**: each joint is a full cell like an energy volume, whose size follows its own energy (owner, 2026-11-15). It pushes back (electromagnetism-like), keeping energy volumes from falling into each other; pull squeezes it and its push swells it back. Energy crosses from one cell to another only through a joint's seams; inside a cell it flows freely. A joint pushes harder the more it is squeezed, rising steeply as it shrinks, like a hard wall, so ordinary pull never beats it and only huge pull reaches the limit; past it comes joint collapse: the joint gives way and the two energy volumes fall into each other, less the joint's energy. The merged volume is more packed, pulls harder, and the collapse runs away (a star collapsing into a black hole). The mirror is **volume split**: a volume that has grown too big, covering too much space to hold itself together against neighbours pulling from opposite sides, tears apart, and the tear becomes new joints. The trigger is size, not density: a split keeps density, so a density trigger would split forever. Volumes are only our labels for energy regions, so as energy spreads the number of volumes and joints grows. **Drift from imperfect transfer** (owner, 2026-11-16): every time energy crosses a seam it rubs against the joint's pushing energy, so the transfer is imperfect and the energy spreads a little. Spread-out energy is less concentrated, so its cells grow: that is expansion, with no push and no dark energy. It slows as space empties, because less energy crosses seams. Whether expansion only looks like it speeds up, because clocks in empty space run at a different rate than in dense regions where it is measured (timescape cosmology, Wiltshire), is a test.
9. **Entanglement**: two connections can carry the same signal strength, inverted, so their behaviour is linked through the grid (the old engine had the split, its pointer and whole settling, axiom 7; the fresh one has none of it yet).

Audit answers (owner, 2026-11-16): joints are thin mortar between volumes, holding a small share of the energy, never cells as big as volumes; energy crosses a joint by diffusion bent by the pull toward packed regions (energy attracts energy; replaced simple diffusion the same day); a joint is squeezed when the pull across it beats its push (2026-11-17: its size is always what its energy sets); there is no size at rest: an energy volume only ever pulls and a joint only ever pushes, and empty space does nothing; lumps form over large regions, energy attracting energy, not from single cells; when a joint collapses, its two volumes become one cell at once; joints carry swirl and oscillate; a tear turns the middle of the torn volume's plus quanta into minus to make the new joint, the mirror of collapse; only the too-big volume splits, never its neighbours; a fresh joint from a tear holds the least energy concentration it will ever have, and its concentration rises as the volumes press on it (it makes more quanta and thins); energy crossing a joint stays plus, since plus attracts plus, and only a tear turns plus into minus; a joint collapses only when its energy concentration gets too high, never because it is low on energy or wants more room; a whirl holds its own core open by its spinning, as in any fluid; a cell needs enough Planck boxes to hold a proper whirl, so there is a minimum number of boxes per cell (owner, 2026-11-16); everything inside a cell (pull, resizing, flow, collapse) runs on that cell's local clock.

### Open problems

Entanglement is open (owner, 2026-11-16): the two halves of a split share one state, and what pushes one must be felt by the other, faster than any wave through the grid could carry it (in Delft in 2015 the two sides were 1.3 km apart and measured faster than light covers that), yet it must never carry a usable message; halves that only share properties from the split cap the match at the 50% that real particles beat (they reach 75%); what links them is unknown, in physics as a whole too; lead (owner, 2026-11-16): a **direct pull**, a second kind of pull made at the split and private to that pair, not passing through neighbours; a reaction with one half wakes it and draws the whole energy into one half, so a detector always finds whole energy; to pass it must (1) pick where it settles by each cell's share of the split energy at that moment (its concentration, as a cell is the unit), so a half's chance is the sum over its cells (70/30 split, 70/30 odds), never by which half reacted first; how: each entangled pair has its own oscillation, a shared state that keeps turning, which is not time and separate from the cells' local clocks (owner, 2026-11-16), and at every tick a pointer flips to a cell with that cell's share as chance; the reaction settles the energy where the pointer is (as in Bell's hopping pointers, 1986); this is fixed in principle by the clock, splitter, whirl and energy, the clock starts at the split, but its speed is the unreadable part (owner, 2026-11-16), or timing a detector to the tick would send a message; ticks must come far faster than any detector can be timed, and (2) for a pair of particles, also turn the other whirl to match, since Bell tests compare filter angles; real entanglement is private to its pair too (a fully entangled half cannot share it with a third); how radiation fits is undecided (guess to test in the engine, not a rule: light is a change in the leak travelling outward; a steady leak is drift and sends nothing, as a steady current sends no light, while a leak that speeds up or slows down, for example when a jolt in a joint changes its seam's width, starts a wave. Heat is seams flickering around the seal point: more flicker, more light, and faster flicker, bluer light (as hotter things glow from red to blue); a cold object's seams are firmly open or shut and send nothing); a regular grid picks out directions, so the
grid should be irregular; length must be an internal quantity, not a coordinate; drift and
repulsion must balance attraction so everything does not clump; bound regions (atoms, galaxies)
must resist the drift while empty space expands; the real expansion first slowed and then sped
up, which a constant drift alone does not give; and electromagnetism is about 10³⁶ times
stronger than gravity, so each force needs its own strength. Measured 2026-11-17 (issue #1): the size law acts per cell and does not add up (two cells merged want a sixteenth of the room, a torn cell's halves sixteen times), so collapses chain and torn halves head for the tear size again; energy runs into single volumes, not regions; at the default numbers nothing seals (a joint's squeeze reaches 0.34 against the seal's 1). Second round that day: a joint cut by a tear kept its squeeze, but two joints joined by a collapse added their sizes, so a volume that tore and fell back together got its joint back 16 times bigger each time (1.9e4 to 1.4e42 boxes in ten steps); the old engine then made joined joints keep their pooled squeeze, and a joint found past its limit collapse instead of standing still. What stays is the size law: a joined joint is 16 times more packed for its size, so in a squeezed web one collapse sets off its neighbours (281 collapses in 19 steps, 73 of them in the last), and the first fitted universe, 22% of whose joints start past the limit, is one volume after 4 steps. With the seal at 0.1 and ten times the pull the whole web is squeezed except in the emptiest places, and volumes seal one by one (839 of 1,728 at most, 64% of the energy) but never a large region (at most 21 volumes together, 1.6%), and they open again: the swing has no friction, so it never comes to rest (its speed went from 1.3 to 37 in 950 steps).

### How results are trusted

**Light on compute** (owner, 2026-11-18, hard rule): a fit is earned, not routine: at most one a night, started only on the owner's say or once a phase's scenes already work by hand. Never more than one run of this project on the graphics card at a time. Every test and probe is the smallest grid and fewest steps that can show the thing, seconds not minutes; think from the code and one measurement before running another; no sweeps to see what happens. The test suite stays quick.

**The fit goes in phases** (owner, 2026-11-18; this replaces fitting on expansion and structure first). Phase 1 (layer 2 of the fresh engine) fits only for gravity to work somewhat normally: two lumps fall toward each other as bodies, sideways motion keeps going, the pull fades as one over distance squared, one lump can go round another. Only when that holds is the next layer added (expansion, structure, black holes, quantum), one at a time, each fitted on top of the last without breaking the layers below. No scorecard and no result pages until phase 1 holds; results are shown in the viewer.

Validation comes first: a measurement is trusted only after it gives the known answer on a
known start (the old engine had the 1/r pull test and the power-spectrum slope test; the fresh one has none yet); the point-scattered
reference spacetimes went with the milestone code and must be rebuilt on the grid of cells before
dimension or proper time is measured. Every result is reported as a
number beside the known value it should match, with the run's seed and settings, so anyone can
rerun it. Misses are reported. Settings may be fitted to measurements, but only on one set and tested on another, as a scorecard of scenes (owner): fit on the shape of the real expansion (slowing, then speeding up) and on structure (lumps forming a web of clumps and threads); test, unfitted, on orbits following Kepler's rule (period squared grows as distance cubed), a bound lump keeping its size while space expands, a black hole's sealed region growing in proportion to its energy, the gap between early and late expansion speed (the Hubble tension, 67.4 against 73.5 km/s per megaparsec), and whether expansion measured with local clocks from inside a dense region looks like it speeds up, as timescape cosmology claims (a 2025 supernova analysis favours it; the Dark Energy Survey disfavours it with other data combined). A setting counts as good only if it does well on the scenes it never saw. The old engine's fit measured space inside a wrapping universe (the sum of its cell lengths, and the summed lengths of the same cells between lumps), never at an edge; its score, written down before a search, was 0 for a universe where nothing happens and about 0 for one that fell into a single cell or tears without end; a setting counted only if its universe still stood in a run four times longer (2026-11-17: nothing fitted). The fresh engine has no fit yet. Every scene runs with all rules on, because the rules depend on each other; a rule is switched off only in a side-by-side comparison run (owner, 2026-11-14). Nothing is placed by hand: scenes start from energy that is slightly uneven everywhere, and lumps must form by themselves. The sim's grid wraps round: there is no edge and no empty space, and the size of the universe is the sum of its cell sizes, which grows or shrinks purely from where the energy sits.

### Milestones

1. **Validate**: measurements recover known results (dimension, proper time, twin paradox).
2. **Grid dynamics**: the energy-length, drift, attraction and repulsion rules run.
3. **Gravity test**: with the attraction rule off, do lumps of energy still move toward each other from the grid's shape alone?
4. **Expansion test**: does drift give accelerating expansion while dense clusters stay bound?
5. **Black hole test**: a collapsed core shrinks, but the sealed region around it (joints past the seal point, nothing leaking out) is the black hole; its radius, measured along the structure, should grow in proportion to the collapsed energy, as relativity's horizon does.
6. **Space test**: does the grid settle into something that behaves like 3D space?
7. **Quantum**: first step, in the fresh engine: every Planck box holds whole quanta and a phase, and one swirl turns in whole steps; spin as an unbreakable whirl is untested. The old engine opened a split by itself where outflow met a fork, with its pointer and whole settling; the fresh one does not yet. Pairs of whirls and filter angles come later. Lead: different kinds of vibration of the same grid (volumes swelling and shrinking, joints pushing back and forth, energy sloshing along seams) are the different fields, each with any number of frequencies. Scale and entanglement (owner, 2026-11-16): an energy volume is about the size of a particle; what others call a particle is a change in the energy of a cell, the energy behaving as a frictionless quantum fluid inside it, not a thing of its own; since cells stay put, a moving particle is that change handed from cell to cell. Energy has no identity, so there are no things that must stay in step: entangled particles are two excited cells far apart that are parts of one state of the whole grid, and when one settles, the whole state settles. The engine stores state box by box and cannot yet hold one state spanning two distant cells; that is the missing piece (the grid as a superposition of configurations). Leads (owner, 2026-11-16): the string of string theory could be a chain through the grid, energy volume, seam, energy volume, with the energy sloshing along it; different sloshing patterns would be different particles and the joints' push the string's tension. The phase in each box is an angle, a position on a tiny circle, which acts like a fifth dimension curled up at every point (Kaluza and Klein, 1921, who got electromagnetism this way); several independent swirls per cell would be several curled dimensions, and the known forces set the target: electromagnetism needs 1 independent direction, the weak force 3, the strong force 8. Lead for spin (owner, 2026-11-16): a real electron needs two full turns to come back the same, which the owner ties to oscillation. In Dirac's belt trick an object tied to its surroundings looks the same after one turn, but the twist in its ties only undoes after a second; a whirl tied into the joint web might give spin one-half this way, without two numbers per Planck box. Test: turn a tied whirl once and twice and check whether the web's twist returns only after two. Lead for charge (owner, 2026-11-16): charge is not a separate thing but the swirl's spikes, like the teeth of a spinning gear: spin is which way the swirl turns and can flip, charge is how many whole spikes the phase winds and which way, which never changes (a magnet flips an electron's spin without touching its charge). Whole spikes give charge in exact equal units; charge is born in opposite pairs at a split (light turning into an electron and a positron), and its push and pull travel as changes of that pattern at light speed, which is light; an electron might be 3 spikes and quarks 1 or 2, matching their 1/3 and 2/3 charges. Lead: the grid as a superposition of configurations, with each connection holding a few numbers rather than one, so several fields are different ways the same grid moves (Levin and Wen, string-net condensation, 2005). There the quantum part is assumed, not derived, and gravity does not emerge; the geometry rules here would supply it.

Where it stands, 2026-11-19, the leak and the crossing parted and the collapse rebuilt (the newest; what follows it is the build before). **What moves energy across a joint**: the two facing boxes meet as equals, at what both hold, and energy goes only the way the fluid there moves, from the phase that stands ahead across the joint; the pull turns that step toward the deeper side and the fluid's own motion carries it. Two volumes that only hold different amounts exchange nothing. **The leak** is its own thing and always happens: each box of a whirl sheds its share toward the direction its phase points to; at a free place it gathers and tears off, toward a neighbour it passes the joint between (toward an edge or corner, one of the sides that way) and arrives at rest, spread once round a core, whatever the neighbour holds (4,572 quanta a turn from a volume of 1.5 calm volumes, to the quantum, beside a neighbour of a half, 1.5 or 6). Measured (6 across): whole whirls 113 of 118 volumes at three layers, 779 of 920 at six (271 turns), 1,002 of 2,106 at eight, 181 of 3,014 at nine (717 turns); through eight layers the inner layers stay whole and the misses are thin outer volumes (833 of the 3,014 hold less than a least whirl), at nine the inner three layers have lost theirs too and why is not found; of the joints few are whole (45 of 2,431). The first volume keeps its energy (1,616 calm volumes at three layers, 1,152 at nine), so the universe grows by the leak alone, about a calm volume a turn: layers at 2.5, 20, 53, 103, 175, 271, 394, 534 and 717 turns, where three took 6.5. A region made 4 times fuller keeps 0.98 of what it was given after half a turn (0.24 before) and 0.83 after four, but does not gain; two such regions do not draw together (2.27 volumes apart, 2.34 after four turns: their energy falls into the first volume between them); one set going moves 0.005 of a volume in a turn. In the bench the fuller volume loses a tenth of a percent a turn by the leak; set going (key M) it gives 0.014 of a calm volume in a turn, five times that, and the fluid arrives moving. **The collapse limit**: the top speed is one speed for the whole universe, 110 calm lengths a turn of a calm swirl, and a volume is past the limit where twice the pull at it is at least that speed squared (a pull of 6,050). The number is chosen: the engine hands the pull on a cell a step, which is no one speed, and under 100.2 the first volume in a full grid (a pull of 3.03 times its energy) would itself be past the limit and never leak. The limit is used twice: a joint whose two volumes are both past it flips to plus where it stands, and out of a volume past it nothing leaves by the leak or the crossing. A collapse changes one cell, the joint: every cell keeps its place, its own whole number (64 bits now) and its other joints, so nothing is pooled and nothing overflows. In `--scene grow` no joint collapses in nine layers, and none can: nothing grown is as deep as the first volume (4,411 at most). `--scene hole` stages one, the first volume and the six at its sides each packed to a first volume's quanta: the first joint falls 18 steps on, six joints fall and no more (13 cells), the joints two cells out stand under 1% from where they stood without the collapse, the hole holds 12,425 calm volumes a turn on and 12,429 after twenty, and with the twelve volumes beside it made fuller it takes them in and stands at 49 cells and 20,390. Size against energy: 13 cells at 12,429, 49 at 19,443, 49 at 23,929: it grows in jumps, not in step. Not right: by this limit a grown universe can never hold a black hole unless the first volume is one (the owner must rule); a merged volume's cells do not mix their fluid; the leak favours some directions by a tenth; the squeezed joints make minus without end (5.6 billion quanta by 156 turns, five times the universe's plus).

Where it stands, 2026-11-19, the whirl rebuilt (the build before; the numbers further down were measured on the whirl before it). A cell is three ring whirls, one for each direction: the fluid goes up through the cell's middle, out over the side that direction points to, back round the outside and in again, round an empty core that is a ring closed inside the cell (no box lies there; the boxes close round it). One ring has an axis and can mesh with a neighbour only at its two ends, so a cell holds three, alike box for box, joined where their jets meet in the middle. Every box holds the same and all of it moves. A calm cell turns once in 25 steps at 6 boxes across; a cell turns in step with its quanta (0.126, 0.252, 0.504 and 1.008 radians a step at a half, 1, 2 and 4 calm volumes) and is smaller by the same law, so where two cells touch their fluid moves alike whatever they hold, as a small gear meshes with a big one. A joint is a volume in a mirror: it turns the other way, and its caps hold the very phases of the volumes' caps it faces (agreement 1.0; 0.99 after 300 steps of an uneven bench); two volumes with no joint between meet half a turn apart, one cap moving out and one in (-1.0). Crossing energy is turned by the joint's own whirl. Nothing is drawn by chance any more: the odd quantum goes where the swirl's own phase stands, and the same start gives the same run. A lone cell holds its whole turns for 10,000 steps (0.7 of counting noise off its rule). Not right: all three rings drawn together are a solid red ball, the core shows only with one ring drawn alone; two uneven volumes stop evening out (3 to 1 ends at 1.9 to 1, the facing ring left partly standing); in the grown cluster few volumes are whole whirls still (2 of 85 at three layers, 52 of 195 at six, against 99 of 1,827 before; nine joints in ten are whole), and the cause is found: with the crossing switched off every volume stays whole (27 of 27), with it on the first volume gives a thin fresh neighbour a third of itself in one turn where the leak's law allows it six parts in ten thousand, and that flood spreads both ways round the ring it enters and stops it. Whether the crossing must obey the leak's share is the owner's to rule. The first volume now keeps most of its energy (1,044 calm volumes at three layers), collapses start at 7 turns and run on, the least whirl is kept at a sixtieth of a calm volume without a derivation, a packed volume's leak is uneven over a turn (3 to 1), and the cluster's picture is 27% off the engine's distances at three layers. Before the rebuild: layers 1, 2, 3c, 4 and 5 of the fresh engine (the numbers of layers 3c and 4 below were measured under the leak and distance laws that layer 5 replaced). Nothing lays a grid, and the tear-in-two is gone. A cell's number is its place in the pattern, three whole numbers, volumes at the even steps and a joint at the odd step between its two: bookkeeping for what the leak has made, never space. `python -m physics.view --scene grow` starts from one energy volume of 2 to the power 30 quanta (1,657 calm volumes; the owner has not said how much). Its whirl leaks: in a turn of its swirl a volume leaks a share of itself that grows with its size (size in calm volumes; the law is under layer 5), alike toward the 26 directions round it. Toward a volume the leak is the crossing through the joint between them. At a free place the leak gathers, and once seven least whirls are there it tears off as a new energy volume: a seventh flips from plus to minus for each side where a volume stands, and that is the joint, plus lost equal to minus gained to the quantum; the pull then squeezes it as before. Joints lie only between volumes that share a side, so of a layer's 26 volumes six have a joint to the volume inside. Measured at layer 3c, before the pull spread (seed 1, 6 boxes across): the first layer, 26 volumes and 54 joints, stands whole after 3.1 turns; then a layer comes about every 0.6 turn, the next starting before the last is full: 483 volumes at 5 turns, 4,095 at 10, 8,894 at 20, 13,895 at 42, 18,746 in 17 layers at 80, still growing and ever slower. That is not exponential: the count grows fast and then slows, and the volumes' sizes summed grow a little faster than in step with time (58,500 calm lengths at 16 turns, 320,000 at 40, 1,030,000 at 80). Nothing stops it but thinness: the outer volumes hold too little for their leak to gather seven least whirls quickly (a quarter of the plus left lies gathered at free places). The first volume's energy floods outward through the fresh joints within a third of a turn, so at 5.5 turns the layers are even (2 calm volumes' quanta each); later the middle stays denser, 0.11 against 0.01 at the rim at 80 turns. Energy never stops moving: 4.6% of the plus changes volume a step after the first layer, 0.06% at 20 turns. The joints make far more minus than the tears flip: 1.07 billion plus becomes 0.49 billion plus and 3.6 billion minus. Not right: the flood breaks every whirl (none of 172 volumes winds its whole turns at three layers; a strict xfail holds the gap); they wind whole again from the middle outward by 11 turns and all do in the inner nine layers at 80, while the outermost volumes have leaked below a whirl's least quanta and do not; as a share a bigger volume leaks more, in quanta it leaks less; a volume is far smaller than its fresh joints, so the picture is red dots in blue until the joints are squeezed; the room of places is a cube made at the start, and a universe that fills it stops; the growth was run at 6 boxes across only; a joint is pinched from the first step and never swings back. `--scene crossing` and the one ball run as before. What is assumed is listed in the `CLAUDE.md` of `axiom_3_structure`, `axiom_4_crossing`, `axiom_5_packing` and `axiom_6_limits`. **Layer 4, the pull**: one number a cell. A volume's pull is its own plus energy, in calm volumes, and a sixth of the pull of the joint at each of its six sides; a joint's is the mean of its two volumes'; each reads its neighbours as they were a step ago, so the pull goes one cell a step, and what would be handed past the rim is lost. The law has no setting. It does two things. A joint is squeezed by `PULL` (2.92) times the pulls at its two volumes over their distance squared, in place of the pull of two neighbours alone. And where the pull differs across a joint by d, the two fluids facing each other there are turned by `BEND` (a quarter turn a swirl turn) times d over 1 plus d, the deeper side back, so more energy crosses toward the deeper pull and keeps that turn as its motion. Measured (seed 1, 6 across): one fuller volume's pull falls a little faster than one over distance (4.8, 2.2, 1.2, 0.78 at 1 to 4 volumes) and its step from cell to cell, which is what turns energy, as one over distance squared (power 2.0 to 2.2); its first trace travels a cell a step, but it builds up like heat, to half its strength two volumes away after 59 steps and four away after 179. The grid no longer stays even: the spread of the volumes' energy (deviation over mean) is 0.29 at 8 turns, 0.79 at 12 and 1.27 at 16, against 0.22 to 0.26 with the pull off; the one gathering is the middle, whose fullest volume holds 1.4, 1.7 and 3.4 calm volumes while the mean falls from 0.58 to 0.21 (0.30 at most at 16 turns with the pull off). Every volume's pull adds up at the middle (185 there against 2 at the rim at 24 turns), so its joints are 8.6 thick at 8 turns, 0.95 at 12, 0.087 at 16 and 0.04 from 20 on, where their push answers the pull, against 46, 31, 18 and 13 at the rim; the reds there stand 12.6 apart at 8 turns and 0.45 at 16, and from 14 turns on they are wider than they stand apart: they touch. Growth goes on layer by layer but slower: 6,433 volumes at 20 turns against 8,894, twelve layers at 18 turns and no thirteenth by 24, as the rim's joints close (nine tenths of what meets in the passage there is minus). **Layer 5, the leak, the distance and joint collapse** (seed 1, 6 across). The leak: in a turn a volume leaks the share size over one plus seven sizes: in step with its size while it is packed, and never a seventh, one of its seven parts (the seventh is assumed; the owner has not ruled it). The distance between two joined volumes' middles is half the one's width, the joint's own size and half the other's, so volumes cannot overlap: the smallest distance through the run was 1.11 times the two half widths. `PULL` stays 2.92, though its reason was stated under the old distance law (by that reason it would now be 278): the owner must rule. A joint collapses when twice the pull at it is at least the top speed squared; the top speed is the pull's own, the joint's distance in two steps (volume, joint, volume), with the pull in calm volumes, the distance in calm lengths and the time in turns of a calm swirl. No number is set, but it reads a calm volume's pull of 1 as a speed squared of 1, which the owner must rule, and it depends on the boxes across because a step does. The joint's minus flips to plus, the same amount, and it and its two volumes are one volume at the fuller one's place, laid as a fresh whole whirl; it keeps every other joint, and joints on one of its six sides share that side. Measured: growth is slower (7 layers and 1,864 volumes at 13.9 turns). Every volume holds at least 2.9 least whirls through the whole run. The first joint falls at 13.86 turns at the centre, where the joints are 0.34 thick and the pull is 150; in 1.6 turns 3,346 joints fall and one volume takes 1,253 places, the inner six layers, holding 2,146 calm volumes' energy at most (1.16 billion minus flipped to plus); it stops at the outer layers, whose joints are fat. Not right: the merged volume does not hold. Its energy crosses back out through the joints it kept (4.7 calm volumes left at 18 turns), the cluster grows again to 10 layers and 4,175 volumes, and a second collapse starts at 23.6 turns; at 23.75 turns its merged volume would hold more than a cell's whole number can (2 to the power 31) and the run stops with that message. The collapsed region does not grow in step with its energy: 903 places for 2,602 calm volumes (a start of 829), 1,253 places for 2,146 (a start of 1,657); a start of 414 filled the whole number after 144 places. With collapse off (a comparison run) every volume stays above a whirl until 30 turns and 9 layers; then the outer ones, behind joints squeezed nearly shut, leak below it (8 at 35 turns, 796 of 3,429 at 70) and growth stalls at 10 layers, so the cascade alone does not carry to 12. Hardly any volume winds its whirl's whole turns (99 of 1,827 at 13.5 turns; a strict xfail). A merged volume leaks only from the one place it stands at, a tracer in the window does not follow a merged volume's far joints, and the same run on the card and on the CPU was not compared. Not right in layer 4: a region made 4 times fuller by hand in a fresh patch keeps 2% of what it was given after half a turn, with the pull or without: through thin fresh joints the crossing's own turn is half a radian a step and the bend at most a thirtieth, so evening-out wins. In the 12-turn cluster such a region keeps 7% and then gains slowly, as the whole inner part does (a region three volumes out gains 49% in five turns); two such regions do not move toward each other cell by cell (5.7 volumes apart, then 6.1), though the room between them shrinks 12% more than without them; a region set going moves one volume in half a turn and stops, because each joint's friction takes the motion. The pull builds up like heat and not as a wave; clocks do not run slower yet; which side a leaked quantum goes to is still a seeded draw, not the swirl's own oscillation. Strict expected failures hold the three staged gaps. What the old engine taught, for the layers to come: nothing but the pull held a lump together, and a pull strong enough to hold a lump held it in the cells it was in, so lumps gathered but did not fall or coast as bodies; crossing took all direction off energy; the size law acts per cell and does not add up (two merged cells want a sixteenth of the room), so collapses chained and emptied volumes grew without limit; the pull fell close to one over distance squared; and no setting in 2,332 fitted universes gave lumps that form and hold.

Success tiers: (1) known physics (falling, bending light, expansion) from the rules; (2) space
and gravity emerge without being forced; (3) a testable prediction that holds, such as the
measured strength of dark energy.
