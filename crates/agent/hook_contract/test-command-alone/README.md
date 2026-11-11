# Crochet « test-command-alone »

`test-command-alone.mts` dit au modèle pourquoi une commande composée ne sort pas du bac à sable MXC.
Dans le banc, quelques commandes sortent du bac par préfixe exact (`ZED_EVAL_MXC_UNSANDBOXED`, préfixes
séparés par `;`). Composée (`git add … && <préfixe>`, ou `<préfixe> > f.log`), la commande reste dans le
bac, où node ne démarre pas (code 3221225794, 0xC0000142). Le 11/11 (run N11111529-RDPD-a, requêtes 36
et 44), l'agent a passé 18 requêtes à chercher un antivirus.

## Le brancher

```json
{
  "agent": {
    "hooks": {
      "PreToolUse": [{ "hooks": [{ "type": "command", "id": "test-command-alone", "command": "node", "args": ["<chemin>/test-command-alone.mts"] }] }]
    }
  }
}
```

Il faut node 24 ou plus, sans aucune dépendance. L'identifiant doit être unique : deux crochets au même
identifiant éteignent tous les crochets (« Agent hooks are not loaded »).

## Ce qu'il décide

- Il lit la liste des préfixes dans `ZED_EVAL_MXC_UNSANDBOXED`, la variable que lit Zed
  (`mxc_unsandboxed_prefix`, `crates/acp_thread/src/terminal.rs`) : aucune copie à tenir à jour. Sans la
  variable, il ne décide rien.
- Il refuse une commande `terminal` qui contient un préfixe sans s'y réduire : elle ne commence pas par
  lui, ou elle porte `;`, `&`, `|`, `` ` ``, `$`, `(`, `)`, `<`, `>` ou un retour à la ligne. Le refus dit
  de lancer le préfixe seul, le reste en commandes séparées, et pourquoi.
- Il ne laisse rien sortir de plus : la liste du bac n'est pas élargie.

## Ce qu'il laisse passer

Toute autre commande, tout autre outil, une entrée illisible. Il ne rejoue pas le contrôle du répertoire
de travail de Zed.
