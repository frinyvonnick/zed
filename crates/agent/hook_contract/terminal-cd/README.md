# Crochet « terminal-cd »

`terminal-cd.mts` répare le champ `cd` de l'outil `terminal`, que Qwen écrit sous un autre nom. Le
02/11, l'agent a écrit `cwd` sept fois de suite ; Zed a répondu « missing field `cd` », et l'agent a
conclu que son terminal était en panne. Il n'a ni lancé les tests ni lu le diff de la story.

## Le brancher

```json
{
  "agent": {
    "hooks": {
      "PreToolUse": [{ "matcher": "terminal|sandboxed_terminal", "hooks": [{ "type": "command", "id": "terminal-cd", "command": "node", "args": ["<chemin>/terminal-cd.mts"] }] }]
    }
  }
}
```

Il faut node 24 ou plus, sans aucune dépendance.

## Ce qu'il fait

- Un champ `cwd` ou `c` devient `cd`, avec la même valeur. Le schéma de l'outil refuse tout champ en
  trop : le champ mal nommé disparaît.
- Sans aucun répertoire, il pose `cd: "."`, la racine du projet.
- Un appel qui a déjà `cd`, un autre outil ou une entrée illisible : aucune sortie, Zed lance l'appel
  tel que le modèle l'a écrit.
- Il ne décide rien : ni accord, ni refus, ni arrêt.

Les tests : `node --test terminal-cd.test.mts`.
