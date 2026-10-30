# Crochet de profils d'agents

`agent-profiles.mts` applique à chaque agent de Zed son profil : son prompt, ses outils, son modèle
et les skills qu'il a le droit d'utiliser. Le format des profils est celui du contrat
(`contrat/profils/README.md`) : un fichier Markdown par agent, avec un en-tête YAML puis le prompt.
Zed ne sait rien des profils : le script passe par les événements génériques des crochets.

## Le brancher

Un seul crochet de commande, déclaré sur trois événements, avec le même `id` pour partager son
dossier de données :

```json
{
  "agent": {
    "hooks": {
      "BeforeModel": [{ "hooks": [{ "type": "command", "id": "agent-profiles", "command": "node", "args": ["<chemin>/agent-profiles.mts", "--agents", "<dossier des profils>"] }] }],
      "PreToolUse": [{ "hooks": [{ "type": "command", "id": "agent-profiles", "command": "node", "args": ["<chemin>/agent-profiles.mts", "--agents", "<dossier des profils>"] }] }],
      "SubagentStart": [{ "hooks": [{ "type": "command", "id": "agent-profiles", "command": "node", "args": ["<chemin>/agent-profiles.mts", "--agents", "<dossier des profils>"] }] }]
    }
  }
}
```

Il faut node 24 ou plus (TypeScript lancé sans compilation), sans aucune dépendance.

## Ce qu'il fait

| Événement | Effet |
|---|---|
| `BeforeModel` | écrit la section `## Agent Profile` (le prompt du profil, puis la liste des profils lançables si `spawn_agent` est offert), retire les outils hors profil (`remove_tools`) et cache les skills non permises (`remove_skills`) |
| `PreToolUse` | refuse un outil hors profil, une skill non permise (outil `skill`, ou `read_file` du `SKILL.md` d'une racine de skills), `create_thread` pour tout sous-agent ; sur `spawn_agent`, refuse un libellé sans profil permis, sinon marque le libellé `[profil#xxxx]` et pose le modèle du profil |
| `SubagentStart` | note le profil du sous-agent dans le dossier de données |

La conversation racine prend le profil `pilote`. Un profil illisible ou ambigu restreint plus, jamais
moins ; dans le doute, il est rejeté :

- l'en-tête est rejeté pour une ligne que le lecteur ne sait pas placer, une table écrite `{...}`,
  une clé entre guillemets ou écrite deux fois, ou un type inattendu (`zed` et `zed.model` doivent
  être des tables ; `zed.tools`, `zed.disallowed_tools` et `skills` des listes) ;
- sans `pilote` lisible, l'agent principal ne garde que les outils de lecture (`read_file`, `grep`,
  `find_path`, `list_directory`), et chaque requête lui rappelle quel fichier corriger ;
- un sous-agent dont le profil est inconnu (renommé, supprimé, illisible, ou lancé avec un jeton que
  le crochet n'a pas émis) voit tous ses outils refusés ;
- si le script échoue, il ferme : l'appel est refusé, ou seuls les outils de lecture restent.

Le jeton `[profil#xxxx]` est noté au `PreToolUse` qui accepte le lancement ; `SubagentStart` ne
reconnaît qu'un jeton émis, et le consomme. Un libellé fabriqué ne donne donc aucun profil.

Chaque raison de rejet d'un profil, et chaque champ pas encore appliqué, s'écrit une fois par contenu
sur la sortie d'erreur, que Zed journalise.

## Fichiers

| Fichier | Rôle |
|---|---|
| `agent-profiles.mts` | le crochet |
| `agent-profiles.test.mts` | ses tests : `node --test agent-profiles.test.mts` |
| `test-agents/` | les profils des tests (copies d'anciens exemples du contrat, plus `lecteur`) |
| `contrat/` | la copie du format des profils au contrat : son README, `agent.schema.json` et ses exemples |

## Ce qui reste ouvert

- Une skill non permise reste lisible autrement que par son `SKILL.md` : ses autres fichiers, `grep`,
  ou `terminal`.
- Le profil d'un sous-agent est relu à chaque appel, pas figé au lancement : modifier un profil
  change les sous-agents déjà lancés.
- Pas encore appliqués, seulement signalés : `effort`, `omitClaudeMd`, `autoCompactWindow`, les
  `hooks` d'un profil et `zed.max_spawn_depth`.
- Chaque appel relit tous les profils : environ 75 ms par appel.
- Zed lit `skill_names` dans le catalogue du message système ; si le gabarit change de forme, la
  liste devient vide.
- Le lecteur YAML est réduit : pas d'ancres, pas de scalaires sur plusieurs lignes sans `|` ou `>`,
  pas de tables `{...}`. Ce qu'il ne sait pas lire fait rejeter le profil. Il reste de petites
  approximations de lecture du texte (échappements dans les guillemets, indicateurs de bloc rares).
- Le rappel « pas de `pilote` lisible » est un texte figé.
- Un profil de projet qui remplace un profil personnel, `pilote` compris, n'est signalé que dans le
  journal : l'utilisateur ne le voit pas.
- Un bloc `<available_skills>` écrit dans les règles de l'utilisateur serait lu par Zed comme le
  catalogue.
- Les fichiers du dossier de données (profils des sous-agents, jetons émis, empreintes des
  avertissements) ne sont jamais nettoyés.
- Si node lui-même ne démarre pas, Zed continue sans restriction : `onFailure: "block"` n'est pas
  encore pris en charge.
