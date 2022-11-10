# Dicionário da documentação

| Nome | Descrição |
| ------ | ------ |
| [🔎](#android_apps_metadatacsv)android_apps_metadata.csv | Arquivo CSV contento dados do Dataset original. |
| [🔎](#top_games_googleplaycsv)top_games_googleplay.csv | Arquivo CSV contento os dados tratados e filtrados. Gerado em tempo de execução no _startup_ |


---
## android_apps_metadata.csv

- Dataset original, sem filtros/alterações. 
- Contém jogos e aplicativos, de diversas fontes diferentes (e não apenas Google Play). 
- Contém 33 colunas (e não apenas as 16 que serão utilizadas).
---

## top_games_googleplay.csv

- Dataset tratado, filtrado por apenas jogos e fonte apenas "google play". 
- Selecionadas apenas as 16 colunas escolhidas para escopo do projeto.
- Removidas linhas com valores inválidos.
- Formatação e correção dos tipos de dados.
---