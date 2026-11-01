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

- Il refuse un `read_file` sans `start_line` ni `end_line` sur un fichier de plus de 120 lignes.
- Il refuse toute fenêtre de plus de 120 lignes. Une fenêtre à une seule borne se mesure comme Zed
  la lit : de `start_line` à la fin du fichier, ou du début à `end_line`.
- Le refus revient au modèle comme résultat de l'outil. Il dit le nombre de lignes et quoi faire :
  chercher d'abord (`grep`, ou un outil de recherche de symboles s'il en a un), puis lire une fenêtre
  de 120 lignes au plus autour des lignes trouvées.
- La limite est la constante `MAX_LINES`, en tête du script.

## Ce qu'il laisse passer

- Un fichier de 120 lignes ou moins, lu en entier (un plan, une story, un petit fichier).
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

## Tests

`node --test read-window.test.mts` lance le vrai script, comme Zed, avec son entrée sur stdin.
