# VYRA GEOCORE — Checkpoint 2026-10-03 — Radiometric Equivalence

## Estado

Checkpoint criado antes da correção do contrato radiométrico e do rerun do Gate 3H.3R.

### Objetivo do pipeline

Construir a base de treinamento do GEOCORE para classificação de cobertura/uso do solo:

- Sentinel2GlobalLULC v2.1
- 194.877 amostras
- 29 classes
- coordenadas CSV → Sentinel-2 real via STAC
- patch 224×224
- bandas B02/B03/B04/B08/B11
- features espectrais + NDVI/NDWI/NDBI
- treinamento posterior com LightGBM

## Evidência decisiva — Earth Search × Planetary Computer

Comparação realizada no mesmo produto, data, tile e baseline 05.11:

| Classe | ES B03 | PC B03 | Δ | ES B08 | PC B08 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 21 | 139 | 1139 | +1000 | 22 | 1022 | +1000 |
| 6 | 510 | 1510 | +1000 | 2154 | 3154 | +1000 |
| 26 | 1432 | 2432 | +1000 | 2577 | 3577 | +1000 |

Itens:

- Classe 21: S2C_30VXM_20250712_0_L2A / S2C_MSIL2A_20250712T112141_R037_T30VXM
- Classe 6: S2B_35NNJ_20241108_0_L2A / S2B_MSIL2A_20241108T083049_R021_T35NNJ
- Classe 26: S2C_20HMK_20251204_0_L2A / S2C_MSIL2A_20251204T140711_R110_T20HMK
- Baseline: 05.11 nos três casos.

### Interpretação operacional

A evidência mostra:

DN_EarthSearch = DN_PlanetaryComputer - 1000

Logo, para o COG do Earth Search usado pelo GEOCORE, o offset de -0.1 não deve ser aplicado novamente como se o DN ainda estivesse no referencial original.

Contrato candidato para validação:

reflectance = DN_EarthSearch × 0.0001

Sem aplicação adicional de -0.1.

## Gates

- 3G.1 STAC recovery: PASS
- 3G.2 metadata contract: PASS como descrição de metadata; equivalência DN↔offset agora resolvida empiricamente
- 3G.3 validity/spatial: PASS
- 3G.4 reflectance/index: PROVISIONAL → precisa rerun
- 3G.5 sanity: PROVISIONAL → precisa rerun
- 3H.1 canonical dataset audit: PASS
- 3H.2 STAC availability: PASS, 29/29
- 3H.3R: PROVISIONAL; execução anterior contaminada pelo offset aplicado duas vezes
- 3H.4 dataset generation: BLOCKED
- ML training: BLOCKED

## Próximo passo obrigatório

1. Atualizar o contrato radiométrico Earth Search.
2. Rerodar o Gate 3H.3R v2 nos 29 representantes.
3. Reavaliar reflectância, índices, SCL e estabilidade.
4. Só então decidir sobre o Gate 3H.4 e a geração em escala.

## Regra de segurança

Não gerar os 194.877 patches nem iniciar treinamento antes do rerun do 3H.3R e da aprovação do contrato radiométrico.
