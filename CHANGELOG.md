# Changelog

## 1.2.0 / 2023-08-28
### What's changed
- Adds Direct Optimization (DPO) style rewards by @opentaco on #99
- Changes print format on exception catch by @camfairchild on #135
- Brings back netuid and wandb to logged config by @p-ferreira on #137
- Adds DPO penalty update by @Eugene-hu  on #138
- Adds original reward output to wandb logs by @isabella618033  on #139
- Reweights reward models by @Eugene-hu  on #140
- Update stale documentation by @steffencruz  on #129



**Full Changelog**: https://github.com/opentensor/validators/compare/v1.1.7...v1.2.0

## 1.1.8 / 2023-08-12
### What's Changed
- Make sure to serve axon first by @camfairchild in 14921d35c
- Adds scripts for releases on github by @camfairchild  in #128
- Wandb config log changes @isabella618033  in #132  

## 1.1.7 / 2023-08-11
### What’s Changed
- Hotfix cutoff limit by @Eugene-hu  in #126

## 1.1.6 / 2023-08-10
### What’s Changed
- Diversity regularization by @isabella618033 in https://github.com/opentensor/validators/pull/124

## 1.1.5 / 2023-08-08
### What’s Changed
- Adds new keywords for the task validator by @p-ferreira in #119
- Save historic embeddings on disk by @opentaco in #121 
- Updates relevance mechanism by @Eugene-hu in #122 

## 1.1.4 / 2023-08-07
- HOTFIX: create and serve the axon at startup by @robertalanm in #120


## 1.1.3 / 2023-08-02
- Adds subtensor to metagraph sync by @camfairchild in #79
- Fix wandb weights format logging by @p-ferreira in #88
- Adds netuid tag to wandb runs by @p-ferreira in #95
- Implements GPU cleaning for optmization by @Eugene-hu in #96
- Adds compatibility with bittensor 5.3.3 by @camfairchild in #107
- Adds historic diversity component by @isabella618033 in #111
- Improvements on diveristy model by @isabella618033 and @Eugene-hu in #111
- Prompt improvements by @mrseeker in #110 and @p-ferreira in #112
- Adds Task Validator Filter to reward pipeline by @p-ferreira in #112
- Fix for empty data retrieval from datasets by @p-ferreira in #113
- Deprecates pip usage by @p-ferreira in #114
