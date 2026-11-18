# Benchmark v2 audit report

## Integrity

```json
{
 "rows": 750,
 "qids": 150,
 "all_5_conditions_per_qid": true,
 "fold_sizes": {
  "0": 30,
  "1": 30,
  "2": 30,
  "3": 30,
  "4": 30
 },
 "every_qid_has_fold": true,
 "comparison_set_size": 20,
 "comparison_subset_of_benchmark": true,
 "validation_errors": 0
}
```

## Dropped questions (6)

Candidates are drawn in a fixed shuffled order (seed 0, 195 spares); a candidate is dropped when its conditions cannot be built cleanly, and the next candidate takes its place. All 5 conditions of a dropped question are removed together.

| qid | question | gold | replacement | reason |
|---|---|---|---|---|
| 57377c98c3c5551400e51efd | What value does the seperating into fermions and bosons depend? | spin / the spin / the value of the spin  | magnitude | counterfactual swap failed: gold string 'spin' still present after word-boundary swap (it occurs inside a longer word/phrase)<br>`…ectrons) have a symmetric magnitude function (e.g. parallel spins) the spatial variables must be antisymmetric (i.e. they ex…` |
| 573750f61c45671900574468 | What kind of measurements define accelerlations? | kinematic / kinematic measurements | Schrödinger equation | counterfactual swap failed: gold string 'kinematic' still present after word-boundary swap (it occurs inside a longer word/phrase)<br>`…can be defined through Schrödinger equation. However, while kinematics are well-described through reference frame analysis in adv…` |
| 57096f37200fba1400367fe7 | Can BSkyB veto the presence of channels on their EPG? | Third-party channels / no | Video On Demand content | counterfactual swap failed: gold string 'no' still present after word-boundary swap (it occurs inside a longer word/phrase)<br>`…t for being carried. However, even in this case, BSkyB does not carry any control over the channel's content or carriage i…` |
| 5728fc9e1d04691400778f12 | When the law is a direct target of the protest, what is this called? | direct civil disobedience | solidarity | counterfactual swap failed: gold string 'direct civil disobedience' still present after word-boundary swap (it occurs inside a longer word/phrase)<br>`…e distinguished between two types of civil disobedience: "Indirect civil disobedience involves violating a law which is not, itself, the object o…` |
| 5729feaf6aef05140015518c | What percentage of the land cleared in the Amazon is used for growing  | 91 / 91% | over half | counterfactual swap failed: gold string '91' still present after word-boundary swap (it occurs inside a longer word/phrase)<br>`…Between 1991 and 2000, the total area of forest lost in the Amazon rose …` |
| 571c7d55dd7acb1400e4c0c6 | What was Dalton's erroneous formula for water? | HO | Pictet | counterfactual swap failed: gold string 'HO' still present after word-boundary swap (it occurs inside a longer word/phrase)<br>`…n 1805, Joseph Louis Gay-Lussac and Alexander von Humboldt showed that water is formed of two volumes of hydrogen and one …` |

## Difference from v1

- questions in both: 145; only in v1: ['57096f37200fba1400367fe7', '5728fc9e1d04691400778f12', '5729feaf6aef05140015518c', '573750f61c45671900574468', '57377c98c3c5551400e51efd']; only in v2: ['56e1febfe3433e140042323a', '5725bad5271a42140099d0be', '573010fab2c2fd14005687db', '57308f6b8ab72b1400f9c584', '57335ddbd058e614000b592f']
- shared questions keep identical C1–C4 passages (only C5 logic changed): 580/580
- same outer fold as in v1: 74/145 (folds are re-drawn by the plan's rule, seed 0, over the new qid set)
- comparison-set overlap with v1: 10/20 (re-drawn by the plan's rule, seed 1)

## C5 counterfactual validation (150 rows) — details in `c5_validation_v2.csv`

- coarse type match: **150/150**
- same question wh-word: **143/150**
- selection tier (1 = same article + wh + form ... 6 = type only): {1: 141, 2: 7, 3: 2}
- swap correct (gold removed everywhere, replacement inserted, passage otherwise unchanged): **150/150**
- rows with grammar flags: **4** ({'plural/singular mismatch?': 4})

Flagged rows (need a human look):

| uid | question | gold → replacement | flags | context |
|---|---|---|---|---|
| 571c3e8cdd7acb1400e4c0a9 | Besides combustion, for what other action did Mayow show nit | respiration → unknown process | plural/singular mismatch? | `… subjects. From this he surmised that nitroaereus is consumed in both unknown process and combustion.…` |
| 57274b35f1498d1400e8f5d7 | Along with private individuals and organizations, what group | religious | religious groups → upper sixth | plural/singular mismatch? | `…hich are run by private individuals, private organizations or rarely, upper sixth and offer a type of education which is…` |
| 572650325951b619008f6fa9 | What was the catalyst that created greater interest in renew | The energy crisis | energy cri → embargo | plural/singular mismatch? | `…Embargo led to greater interest in renewable energy, nuclear power and domest…` |
| 5727f7523acd2414000df110 | What has presented problems to the US economy more than othe | decline of organized labor | t → satisfaction and happiness | plural/singular mismatch? | `…gist Jake Rosenfield of the University of Washington asserts that the satisfaction and happiness in the United States ha…` |
