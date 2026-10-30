# Profils d'agents pour Zed — au format des agents de Claude Code

Un profil d'agent décrit un type d'agent : ses outils, ses skills, son modèle, ses crochets, et son prompt. Yvonnick a choisi de partir du format des agents (« subagents ») de Claude Code, vérifié à la source. Ce dossier donne ce format exact, puis propose son schéma pour Zed.

Toute affirmation sur Claude Code cite le texte brut de sa documentation, enregistré dans [`sources/claude-code-sub-agents-2026-10-30.md`](../sources/claude-code-sub-agents-2026-10-30.md) (<https://code.claude.com/docs/en/sub-agents.md>, téléchargé le 30/10/2026 avec `curl -sL`). Ce que la source ne dit pas est marqué « non trouvé dans la source ». `verifier.mjs` contrôle que chaque citation pointe vers une ligne existante.

| Fichier | Rôle |
|---|---|
| [`agent.schema.json`](agent.schema.json) | le schéma JSON (2020-12) de l'en-tête d'un profil, compatible avec Claude Code, plus les extensions de Zed sous `zed` |
| [`exemples/projet/explorer.md`](exemples/projet/explorer.md) | un profil de projet : un explorer en lecture seule, sans `hooks` |
| [`exemples/personnel/planner.md`](exemples/personnel/planner.md) | un profil personnel : un planner, avec une garde qui ne le laisse écrire que son plan |
| [`exemples/personnel/pilote.md`](exemples/personnel/pilote.md) | le profil par défaut de l'agent principal, avec sa liste blanche de sous-agents |
| [`exemples/personnel/general.md`](exemples/personnel/general.md) | un sous-agent sans restriction d'outils, quand aucun profil spécialisé ne convient |

## Le format chez Claude Code

### Un fichier Markdown avec un en-tête YAML

Un agent est un fichier Markdown. Il commence par un en-tête YAML entre deux lignes `---`, suivi du prompt système en Markdown (`sources/claude-code-sub-agents-2026-10-30.md:246`, `sources/claude-code-sub-agents-2026-10-30.md:293`).

- **Le corps devient le prompt système de l'agent.** L'agent ne reçoit que ce prompt, plus des informations d'environnement comme le dossier de travail ; il ne reçoit pas le prompt système de Claude Code (`sources/claude-code-sub-agents-2026-10-30.md:270`).
- **Seuls `name` et `description` sont obligatoires** (`sources/claude-code-sub-agents-2026-10-30.md:293`).
- Les noms de champ composés s'écrivent en camelCase. **Un champ inconnu est ignoré sans erreur** (`sources/claude-code-sub-agents-2026-10-30.md:295`).

Un fichier est ignoré dans quatre cas (`sources/claude-code-sub-agents-2026-10-30.md:339`) :

- il n'a pas de `name` ;
- le `---` d'ouverture n'est pas sur la première ligne ;
- il a un `name` mais pas de `description` ;
- son YAML ne se lit pas.

Un `name` mal formé l'écarte aussi : il ne doit ni commencer par `-`, ni contenir `:`, ni dépasser 256 caractères.

### Où vivent les fichiers

| Emplacement | Portée | Priorité | Source |
|---|---|---|---|
| réglages gérés par l'organisation | toute l'organisation | 1 (la plus haute) | `sources/claude-code-sub-agents-2026-10-30.md:164` |
| option `--agents` (JSON) | la session | 2 | `sources/claude-code-sub-agents-2026-10-30.md:165` |
| `.claude/agents/` | le projet | 3 | `sources/claude-code-sub-agents-2026-10-30.md:166` |
| `~/.claude/agents/` | tous les projets de l'utilisateur | 4 | `sources/claude-code-sub-agents-2026-10-30.md:167` |
| dossier `agents/` d'un plugin | là où le plugin est actif | 5 | `sources/claude-code-sub-agents-2026-10-30.md:168` |

- À nom égal, l'emplacement le plus prioritaire gagne (`sources/claude-code-sub-agents-2026-10-30.md:160`).
- Les dossiers sont lus récursivement, et le sous-dossier ne compte pas dans l'identité : seul `name` l'est (`sources/claude-code-sub-agents-2026-10-30.md:178`).
- En JSON (`--agents`), chaque clé est un nom d'agent, et le champ `prompt` remplace le corps Markdown (`sources/claude-code-sub-agents-2026-10-30.md:224`, `sources/claude-code-sub-agents-2026-10-30.md:226`).

### Les champs de l'en-tête

| Champ | Type et valeurs | Obligatoire | Ce qu'il fait | Source |
|---|---|---|---|---|
| `name` | chaîne, 256 caractères au plus, sans `:` ; ne commence pas par `-` | oui | identifiant unique ; les crochets le reçoivent comme `agent_type` | `sources/claude-code-sub-agents-2026-10-30.md:302` |
| `description` | chaîne | oui | quand déléguer à cet agent | `sources/claude-code-sub-agents-2026-10-30.md:303` |
| `tools` | chaîne séparée par des virgules, ou liste YAML | non | outils permis ; sans ce champ, l'agent hérite de tous les outils | `sources/claude-code-sub-agents-2026-10-30.md:304` |
| `disallowedTools` | même format que `tools` | non | outils refusés, retirés de la liste ; ils s'appliquent avant `tools` (`sources/claude-code-sub-agents-2026-10-30.md:450`) | `sources/claude-code-sub-agents-2026-10-30.md:305` |
| `model` | `sonnet`, `opus`, `haiku`, `fable`, un identifiant complet, ou `inherit` | non | modèle de l'agent | `sources/claude-code-sub-agents-2026-10-30.md:306` |
| `permissionMode` | `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan`, `manual` | non | mode de permission | `sources/claude-code-sub-agents-2026-10-30.md:307` |
| `maxTurns` | nombre | non | nombre de tours au-delà duquel l'agent s'arrête, avec une sortie marquée partielle | `sources/claude-code-sub-agents-2026-10-30.md:308` |
| `autoCompactWindow` | nombre de jetons, de 100000 à 1000000 | non | abaisse le seuil de compaction automatique | `sources/claude-code-sub-agents-2026-10-30.md:309` |
| `skills` | liste | non | skills chargées dans le contexte au démarrage, au plus les 32 premiers noms distincts (`sources/claude-code-sub-agents-2026-10-30.md:627`). N'empêche pas d'en appeler d'autres | `sources/claude-code-sub-agents-2026-10-30.md:310` |
| `mcpServers` | liste de noms de serveurs, ou de définitions en ligne | non | serveurs MCP de l'agent | `sources/claude-code-sub-agents-2026-10-30.md:311` |
| `hooks` | objet, au format des crochets | non | crochets actifs seulement pendant que l'agent tourne ; un crochet `Stop` devient `SubagentStop` (`sources/claude-code-sub-agents-2026-10-30.md:799`) | `sources/claude-code-sub-agents-2026-10-30.md:312` |
| `memory` | `user`, `project`, `local` | non | mémoire persistante | `sources/claude-code-sub-agents-2026-10-30.md:313` |
| `background` | booléen | non | garder l'agent en arrière-plan | `sources/claude-code-sub-agents-2026-10-30.md:314` |
| `omitClaudeMd` | booléen | non | démarrer sans les fichiers CLAUDE.md | `sources/claude-code-sub-agents-2026-10-30.md:315` |
| `effort` | `low`, `medium`, `high`, `xhigh`, `max` | non | niveau d'effort | `sources/claude-code-sub-agents-2026-10-30.md:316` |
| `isolation` | `worktree` | non | lancer l'agent dans une copie git temporaire | `sources/claude-code-sub-agents-2026-10-30.md:317` |
| `color` | `red`, `blue`, `green`, `yellow`, `purple`, `orange`, `pink`, `cyan` | non | couleur d'affichage | `sources/claude-code-sub-agents-2026-10-30.md:318` |
| `initialPrompt` | chaîne | non | premier message, quand l'agent est l'agent principal | `sources/claude-code-sub-agents-2026-10-30.md:319` |
| `experimental` | objet ; seule clé documentée : `cacheTtl`, `5m` ou `1h` (`sources/claude-code-sub-agents-2026-10-30.md:322`) | non | options expérimentales | `sources/claude-code-sub-agents-2026-10-30.md:320` |

Deux détails de forme :

- `tools` accepte la syntaxe `Agent(nom1, nom2)` pour limiter les sous-agents que l'agent peut lancer (`sources/claude-code-sub-agents-2026-10-30.md:503`) ;
- **Non trouvé dans la source** : une borne pour `maxTurns`, et le schéma exact d'une définition MCP en ligne (la source renvoie à la page des réglages MCP).

## La proposition pour Zed

[`agent.schema.json`](agent.schema.json) accepte tels quels tous les champs ci-dessus, avec leurs valeurs. Un fichier d'agent de Claude Code se charge donc dans Zed.

**Le premier niveau est ouvert.** Claude Code ignore les champs qu'il ne connaît pas (`sources/claude-code-sub-agents-2026-10-30.md:295`) : un fichier écrit pour une version plus récente doit continuer à se charger. Zed journalise ces champs inconnus.

### Les extensions de Zed, sous `zed` (bloc fermé)

| Champ | Pourquoi Zed en a besoin |
|---|---|
| `zed.tools`, `zed.disallowed_tools` | les outils de Zed ne portent pas les noms de ceux de Claude Code (`read_file` contre `Read`). Quand ils sont présents, Zed les applique à la place de `tools` et `disallowedTools`, dans le même ordre : les refus d'abord. Un outil refusé est retiré de la liste montrée au modèle (le filtre d'`enabled_tools` de la page de conception), et pas seulement refusé à l'appel |
| `zed.model` (`provider`, `id`) | un alias Claude (`sonnet`) n'a pas de sens sur un fournisseur de Zed comme vLLM |
| `zed.max_spawn_depth` | la page de conception veut refuser un sous-agent au-delà d'une profondeur. Le refus est rendu au modèle comme résultat de `spawn_agent`, et le modèle continue |

### Ce que Zed fait des autres champs en 0.1

Décisions d'Yvonnick (« Oui je valide ») :

- **Appliqués** : `name`, `description`, le corps (le prompt), `skills`, `hooks` (profils personnels seulement, au format du contrat des crochets, limités aux événements de Zed), `effort`, `omitClaudeMd` (sans les fichiers de règles du projet) et `autoCompactWindow`.
- **Acceptés mais non appliqués en 0.1, avec un avertissement au chargement qui les nomme** : `permissionMode`, `memory`, `background`, `isolation`, `color`, `initialPrompt`, `experimental` et `mcpServers`. Cela fait huit champs, et non sept comme l'annonçait un compte précédent.
- **`maxTurns` : accepté dans le format, ignoré par Zed**, qui l'écrit dans son journal au chargement. Un plafond de tours coupe le modèle, ce que la doctrine du dépôt interdit.

### Où vivent les profils

| Portée | Emplacement | `hooks` permis ? |
|---|---|---|
| personnel | le dossier de configuration de Zed | oui |
| projet | `.zed/agents/` à la racine du projet | **non** : Zed refuse de charger le profil, avec un message qui nomme le fichier |

Le refus des `hooks` dans un profil de projet suit la même prudence que les réglages de projet, qui ne peuvent pas déclarer de crochets aujourd'hui : un dépôt cloné ne doit pas pouvoir lancer de commande sur la machine.

Chez Claude Code, la règle est différente : les crochets d'un agent de projet ne tournent qu'une fois le dossier accepté comme « de confiance » (`sources/claude-code-sub-agents-2026-10-30.md:767`).

L'ordre entre un profil personnel et un profil de projet de même nom n'est pas encore décidé. Chez Claude Code, le projet passe avant l'utilisateur (`sources/claude-code-sub-agents-2026-10-30.md:166`).

## Choisir le profil d'un sous-agent

Décisions d'Yvonnick : « Sauf qu'en pre tool du spawn agent on peut balancer une erreur si y a pas de profil sélectionné ? », « Et par défaut on prendra le pilote pour l'agent principal », et, à propos de la liste des profils : « ça doit faire parti de l'agent pilote non ? ».

### 1. La liste blanche vit dans le profil de l'agent principal

Le modèle vient de Claude Code. L'agent principal y limite les sous-agents qu'il peut lancer par `tools: Agent(worker, researcher)` (`sources/claude-code-sub-agents-2026-10-30.md:503`, exemple `sources/claude-code-sub-agents-2026-10-30.md:513`). C'est une liste blanche : un autre type échoue, et l'agent ne voit que les types permis (`sources/claude-code-sub-agents-2026-10-30.md:517`). `Agent` sans parenthèses permet tout (`sources/claude-code-sub-agents-2026-10-30.md:519`). Sans `Agent`, aucun sous-agent (`sources/claude-code-sub-agents-2026-10-30.md:525`). Chez Claude Code, cette syntaxe ne vaut que pour l'agent principal (`sources/claude-code-sub-agents-2026-10-30.md:527`).

Dans Zed, la même liste s'écrit dans `zed.tools` :

| Entrée dans `zed.tools` | Effet |
|---|---|
| `spawn_agent(explorer, planner, general)` | seuls ces profils peuvent être lancés |
| `spawn_agent` | tout profil est permis |
| aucune entrée `spawn_agent` | aucun sous-agent |

**Cette liste est la seule source de vérité.** Elle sert à deux choses :

- **la montrer** : le crochet de profils écrit la section `## Agent Profile` dans le message système de l'agent principal. Il passe par le champ générique `system_prompt_insert` de `BeforeModel`, et Zed l'insère avant `## System Information` (voir [le README du contrat](../README.md), section « `BeforeModel` : insérer un texte dans le message système »). La section reprend le nom et la description de chaque profil permis, dans l'ordre de la liste, sous une forme toujours identique, condition pour que le serveur garde le début de prompt en cache. Exemple réel : [`examples/BeforeModel.output.agent-profile.json`](../examples/BeforeModel.output.agent-profile.json).

  ```
  ## Agent Profile
  Profils de sous-agent permis (commence le libellé de spawn_agent par [nom]) :
  - explorer : <description du profil explorer>
  - planner : <description du profil planner>
  - general : <description du profil general>
  ```

- **refuser** : le crochet `PreToolUse` sur `spawn_agent` la consulte avant chaque lancement (point 2).

### 2. Désigner le profil au lancement

Le profil se donne en tête du libellé : `spawn_agent` avec `label: "[explorer] où est lu le jeton ?"`.

- **Libellé sans profil, ou profil hors de la liste** : le `PreToolUse` sur `spawn_agent` répond `deny`, avec une raison qui rappelle la liste permise. Exemple : « Commence le libellé par un profil permis : [explorer], [planner], [general]. » Le refus redirige ; il ne coupe pas, et le modèle relance avec un bon libellé.
- **Profil permis** : le crochet répond `allow` et réécrit le libellé par `updatedInput` avec un jeton unique, par exemple `[explorer#a1b2] où est lu le jeton ?`. `SubagentStart` reçoit ce libellé dans `agent_type` ; il en tire le profil à appliquer et l'instance précise.

### 3. Les profils par défaut

- **L'agent principal prend par défaut le profil `pilote`** : [`exemples/personnel/pilote.md`](exemples/personnel/pilote.md).
- **Le profil `general`**, s'il figure dans la liste, permet un sous-agent sans restriction d'outils : [`exemples/personnel/general.md`](exemples/personnel/general.md). Sa description dit de s'en servir quand aucun profil spécialisé ne convient.

### Ce qui reste à trancher

- **Tranché par Yvonnick (« 2 oui »)** : c'est un crochet qui écrit la section `## Agent Profile`, par le champ générique `system_prompt_insert` de `BeforeModel`. Zed ne porte aucun métier de profil dans cette insertion.
- **Le format du jeton** (`#` suivi de quatre caractères) est une proposition.

## Les exemples

- [`exemples/personnel/pilote.md`](exemples/personnel/pilote.md), le profil par défaut de l'agent principal. Sa liste blanche : `spawn_agent(explorer, planner, general)`.
- [`exemples/personnel/general.md`](exemples/personnel/general.md), un sous-agent sans restriction d'outils, pour quand aucun profil spécialisé ne convient.
- [`exemples/projet/explorer.md`](exemples/projet/explorer.md), un profil de projet sans `hooks`. Il est en lecture seule : ses outils sont `read_file`, `grep`, `find_path`, `list_directory` et `skill`, et l'écriture, le terminal et `spawn_agent` lui sont refusés. Il précharge la skill `exploring-code` et ne peut lancer aucun sous-agent (`max_spawn_depth: 0`).
- [`exemples/personnel/planner.md`](exemples/personnel/planner.md), un profil personnel. Il peut écrire, mais une garde `PreToolUse` sur `write_file`, déclarée dans son en-tête, ne le laisse écrire que `docs/plan.md`. Il précharge `slicing-work` et peut lancer un niveau de sous-agents.

`verifier.mjs` contrôle chaque exemple :

- il lit son en-tête YAML et le valide contre `agent.schema.json` ;
- il vérifie que le corps (le prompt) n'est pas vide ;
- il refuse un profil de `exemples/projet/` qui déclare des `hooks` ;
- il vérifie que chaque profil cité dans une liste blanche `spawn_agent(...)` existe parmi les exemples, et que le profil `pilote` existe.
