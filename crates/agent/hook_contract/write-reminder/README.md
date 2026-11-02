# Crochet « write-reminder »

`write-reminder.mts` pousse l'agent à écrire quand il lit sans fin. Le 01/11, un sous-agent qui
devait écrire un seul test a lu 72 fichiers en 70 minutes sans écrire une ligne, et relisait les
mêmes fichiers après chacune de ses trois compactions. Le crochet ne coupe jamais le modèle : il
ajoute un rappel, et rien d'autre.

## Le brancher

Le même script se déclare sur trois événements :

```json
{
  "agent": {
    "hooks": {
      "BeforeModel": [{ "hooks": [{ "type": "command", "id": "write-reminder", "command": "node", "args": ["<chemin>/write-reminder.mts"] }] }],
      "AfterModel": [{ "hooks": [{ "type": "command", "id": "write-reminder", "command": "node", "args": ["<chemin>/write-reminder.mts"] }] }],
      "PostCompact": [{ "hooks": [{ "type": "command", "id": "write-reminder", "command": "node", "args": ["<chemin>/write-reminder.mts"] }] }]
    }
  }
}
```

Il faut node 24 ou plus, sans aucune dépendance.

## Ce qu'il fait

- **Compter les requêtes sans écriture** (`BeforeModel`). Chaque requête au modèle compte pour une.
  Une requête dont les nouveaux messages portent un résultat réussi d'un outil d'écriture
  (`edit_file`, `write_file`, `create_directory`, `copy_path`, `move_path`, `delete_path`,
  `rename_symbol`, `apply_code_action`) remet le compte à un. Une écriture échouée ne compte pas.
- **Rappeler d'écrire** toutes les 8 requêtes sans écriture (constante `REMINDER_EVERY`), par un
  message éphémère : « You have made N model requests since your last write. … ». Quand une taille
  de contexte est mesurée, le rappel la donne aussi.
- **Dire la taille du contexte** au-delà de 70 % de la fenêtre (constante `LEVEL_EVERY_REQUEST`) :
  à chaque requête sans écriture, « Context window: 90k of 123k tokens (73 %). Around 111k, Zed
  compacts the conversation… ».
- **Après chaque compaction** (`PostCompact`), ajouter à la fin du résumé le premier message
  utilisateur du fil, mot pour mot (le brief d'un sous-agent), puis « The facts you read are in your
  summary above: do not read those files again; write the next file now. ». À chaque compaction,
  pas une seule fois.

## D'où vient la taille du contexte

C'est une mesure, pas une estimation. À chaque réponse du modèle, `AfterModel` reçoit le `usage`
que le fournisseur a renvoyé (vLLM pour Qwen). La taille est la somme de ses quatre champs :
`input_tokens + cache_read_input_tokens + cache_creation_input_tokens + output_tokens`, comme
`TokenUsage::total_tokens` de Zed. La conversion OpenAI de Zed retire déjà les jetons lus en cache
de `input_tokens` (`crates/open_ai/src/completion.rs:1485-1488`) : la somme ne les compte qu'une
fois.

- Le chiffre donné est celui de la **dernière réponse**. Les résultats d'outils arrivés depuis n'y
  sont pas encore : le vrai niveau est un peu plus haut.
- Tant qu'aucune réponse n'est mesurée, le crochet ne dit rien de la fenêtre.
- La réponse à la requête de compaction, et celle d'un titre ou d'un résumé de panneau, ne sont pas
  des mesures : la première porte encore l'ancien contexte.
- Après une compaction, la taille est oubliée jusqu'à la prochaine réponse mesurée.
- La fenêtre (`CONTEXT_WINDOW`, 122 880 jetons) et le seuil de compaction de Zed
  (`COMPACTION_THRESHOLD`, 110 592, soit 90 %, lu dans les journaux : « threshold is 110592 of
  122880 ») sont des constantes : à changer avec le modèle.

## Ce qu'il ne décide pas

- Il ne refuse aucun outil et n'arrête jamais le modèle.
- Il ne compte ni la requête de compaction ni les requêtes de titre ou de résumé.
- Il n'appelle jamais le serveur du modèle, et ne relit jamais la conversation : il ne lit que
  `new_messages` (les messages que Zed ne lui a pas encore envoyés) et le `usage` d'`AfterModel`.

## Ce qu'il garde

Un petit fichier par conversation (`zed.thread_id`) dans `ZED_HOOK_DATA_DIR` : le premier message
utilisateur, le nombre de requêtes depuis la dernière écriture, la dernière taille mesurée et le
dernier rappel. Jamais l'historique. Zed rappelle `BeforeModel` quand il relance une requête, sans
nouveau message : le crochet ne la compte pas une deuxième fois et lui redonne le même rappel.

Le but est ce premier message privé des blocs que Zed y ajoute (`<skill_content>` d'une commande
slash, `<system-reminder>`), espaces de tête et de fin retirés ; s'il ne reste rien, le message
entier. Si le crochet est branché au milieu d'un fil déjà compacté, le premier message qu'il voit
est le résumé : c'est lui qu'il gardera comme but.

## Prouver qu'il se déclenche

Chaque message réellement envoyé au modèle ajoute une ligne JSON à
`<ZED_HOOK_DATA_DIR>/<fil>.firings.jsonl` (même nom haché que le fichier d'état) : `at` (heure ISO),
`kind` (`write-nudge`, `context-level` ou `goal-after-compaction`), `requestsSinceWrite`,
`contextTokens` et `percent` (`null` sans mesure), et `text`, le texte exact envoyé. Rien n'y est
écrit quand le crochet ne renvoie rien. Pour une compaction, `contextTokens` est la dernière mesure
avant elle.

## Tests

`node --test write-reminder.test.mts` lance le vrai script, comme Zed, avec son entrée sur stdin.
Un test mesure le temps ajouté sur une requête de 200 messages : moins de 50 ms au-delà du
démarrage de node.
