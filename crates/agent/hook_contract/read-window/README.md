# Crochet « read-window »

`read-window.mts` empêche l'agent de lire un long fichier en entier. Un `read_file` entier reste dans
la conversation et repart à chaque requête. Le 01/11, un sous-agent qui devait écrire un seul test a
fait 78 lectures entières (environ 762 000 caractères) sans écrire une ligne.

## Le brancher

```json
{
  "agent": {
    "hooks": {
      "PreToolUse": [{ "matcher": "read_file", "hooks": [{ "type": "command", "id": "read-window", "command": "node", "args": ["<chemin>/read-window.mts"] }] }]
    }
  }
}
```

Il faut node 24 ou plus, sans aucune dépendance.

## Ce qu'il décide

- Il refuse un `read_file` sans `start_line` ni `end_line` sur un fichier de plus de 300 lignes et
  de 16 384 caractères ou moins.
- Il refuse toute fenêtre de plus de 300 lignes. Une fenêtre à deux bornes se juge sur ce que le
  modèle demande (`end_line - start_line + 1`), sans ouvrir le fichier. Une fenêtre à une seule borne
  se mesure comme Zed la lit : de `start_line` à la fin du fichier, ou du début à `end_line`.
- Pour compter les lignes, il lit le fichier par morceaux de 64 Kio, sans jamais le garder entier en
  mémoire. Sur un fichier de 10 Mo, il ajoute moins de 50 ms au démarrage de node (test).
- Le refus revient au modèle comme résultat de l'outil. Il reprend les mots et la structure de la
  réponse de Zed pour un gros fichier (`crates/agent/src/tools/read_file_tool.rs:522-530`), pour que
  le modèle ne lise qu'une seule façon de le dire : « This file is too large to read all at once… »,
  « IMPORTANT: Do NOT retry this call without line numbers… », « NEXT STEPS: … ». Seul change ce
  qui diffère : pas de plan ici, donc chercher d'abord les lignes avec `grep`, puis lire une
  fenêtre de 300 lignes au plus. Une fenêtre trop grande reçoit le même texte, ouvert par « This
  window is too large to read at once… ».
- La limite est la constante `MAX_LINES`, en tête du script.

## Ce qu'il laisse passer

- Un fichier de 300 lignes ou moins, lu en entier (un plan, une story, un petit fichier).
- Un fichier de plus de 16 384 caractères lu sans fenêtre : Zed n'en renvoie pas le contenu mais le
  plan (ses symboles et leurs lignes), ce qui aide à viser la fenêtre suivante. La limite est
  `AUTO_OUTLINE_SIZE` de Zed (`crates/agent/src/outline.rs:10`), comparée par `>` à la longueur du
  texte tel que Zed le garde, où chaque `\r\n` devient un `\n` (`crates/text/src/text.rs:48`) :
  le crochet compte les caractères de la même façon.
- Tout fichier sous `.agents/skills/` ou `.claude/skills/`, projet ou dossier personnel : une skill
  se lit en entier.
- Un fichier qu'il ne trouve pas, une entrée illisible, un autre outil : aucune sortie, Zed exécute
  l'appel tel quel.

## Trouver le fichier

Zed donne souvent le chemin précédé du nom du dossier racine du projet
(`epic-socle-v2-jetable/epics/...`), ou un chemin absolu. Le crochet prend le dernier segment de
`cwd` comme nom du dossier racine, comme `include-pattern-racine.mts`, et cherche le fichier sous
`cwd`. Il ne connaît que la première racine du projet : un fichier d'une autre racine n'est pas
trouvé, donc pas refusé.

## Prouver qu'il agit

Chaque refus réellement renvoyé ajoute une ligne JSON à `<ZED_HOOK_DATA_DIR>/<fil>.firings.jsonl`.
Zed donne à chaque crochet son propre dossier de données. `<fil>` est le début du hachage SHA-256 de
`zed.thread_id`, ou de `session_id` sans lui, comme grep-court et write-reminder. Chaque ligne porte
`at` (heure ISO), `kind` (`whole-file` ou `window-too-large`), `path` (le chemin donné par le
modèle), `lines` (taille du fichier ou de la fenêtre) et `text` (le motif exact). Rien n'y est
écrit quand le crochet laisse passer.

## Tests

`node --test read-window.test.mts` lance le vrai script, comme Zed, avec son entrée sur stdin.
