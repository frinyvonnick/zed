# Crochet « grep court »

`grep-court.mts` raccourcit les gros résultats de l'outil `grep` dans ce que l'agent envoie au
modèle. Un résultat de plus de 6 000 caractères reste sinon dans la conversation, et repart à
chaque requête. Zed ne sait rien de cette règle : le script passe par `tool_result_replacements`
de `BeforeModel`.

## Le brancher

```json
{
  "agent": {
    "hooks": {
      "BeforeModel": [{ "hooks": [{ "type": "command", "id": "grep-court", "command": "node", "args": ["<chemin>/grep-court.mts"] }] }]
    }
  }
}
```

Il faut node 24 ou plus, sans aucune dépendance.

## Ce qu'il fait

- Un résultat de `grep` réussi de plus de 6 000 caractères devient :
  - ses 12 premières lignes, dont la ligne de pagination quand il y en a une ;
  - une phrase qui dit comment voir le reste (`read_file`) ou obtenir moins de résultats (motif plus
    précis, `include_pattern`) ;
  - la liste de toutes ses correspondances par fichier, du fichier qui en a le plus au moins,
    chacune avec ses lignes : `proxy/src/a.ts (2): L10-14, L40-44`.
- Les autres résultats ne changent pas, ni ceux des autres outils, ni les échecs de `grep`.
- L'historique de l'agent garde le résultat entier : seul ce qui part au modèle change.

## Ce qu'il garde

Zed applique un remplacement à une seule requête, et n'envoie au crochet que les messages qu'il
n'a pas encore vus. Le crochet garde donc dans `ZED_HOOK_DATA_DIR` un fichier par conversation
(`zed.thread_id`), avec ses remplacements, et les répète à chaque requête avec le même texte : le
serveur réutilise ainsi le début de prompt qu'il a en cache.

Il répète un remplacement seulement tant que son message fait partie des messages déjà connus
(`known_message_count`) : Zed rejette un remplacement qui vise un résultat absent de la requête,
par exemple après une compaction.

## Tests

`node --test grep-court.test.mts` lance le vrai script, comme Zed, avec son entrée sur stdin.
