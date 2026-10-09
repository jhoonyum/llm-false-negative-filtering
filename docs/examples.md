# Flagged examples

Five (query, passage) pairs where the verifier answered Yes. Each shows the passage MS MARCO labels as the positive for the query and the BM25 candidate the verifier flagged, which training could otherwise use as a negative.

How these were picked: `extract_examples.py` ranks the 6,794 flagged pairs by query-word overlap and writes the top 20. All 20 contain every query word, a tie shared by 1,118 flagged pairs, and the sort keeps file order within a tie, so the 20 come from the first 45 queries. The five below were picked by hand from that list, one per query, skipping passages about named people. They are not a random sample and say nothing about how often the verifier is wrong.

Passages are quoted from [MS MARCO v2.1](https://huggingface.co/datasets/microsoft/ms_marco) (Microsoft), which is released for non-commercial research use only. They are trimmed; `[...]` marks a cut. Spelling and spacing are as in the source.

---

## 1. `kyphosis causes`

**Labeled positive**

> Causes. Kyphosis may be caused by poor posture during childhood or be the result of abnormally shaped vertebrae or developmental problems with the spine. [...]

**Flagged**

> Conditions that cause kyphosis Causes Kyphosis may be caused by poor posture during childhood or be the result of abnormally shaped vertebrae or developmental problems with the spine. The spine

Nearly the same text as the labeled positive. As a negative, it would push the model away from a passage it should rank first. The verifier said Yes to all five of this query's candidates.

## 2. `what is a tca chemical peel`

**Labeled positive**

> 1 Trichloroacetic acid (TCA) is the main peeling agent used for medium peels, though the peel may also be done in several steps using a different chemical solution followed by TCA. [...]

**Flagged**

> Having been used for more than 20 years, the TCA peel is a non-toxic chemical peel. TCA stands for trichloroacetic acid which is considered a relative of vinegar. [...]

Defines the term the query asks about.

## 3. `legal definition accession`

**Labeled positive**

> Definition of accession. 1  1a : the act or process by which someone rises to a position of honor or power [...]

**Flagged**

> [...] Accede/Accession: ‘Accession’ is an act by which a State signifies its agreement to be legally bound by the terms of a particular treaty. It has the same legal effect as ratification, but is not preceded by an act of signature. [...]

The labeled positive gives the general dictionary sense. The flagged passage gives a legal sense, which is closer to what the query asks for.

## 4. `is the roth ira taxable`

**Labeled positive**

> A Roth IRA is an Individual Retirement Account that provides tax-free growth. [...]

**Flagged**

> Roth IRAs. Roth IRA contributions are not deductible so the distributions are tax free. You've paid tax on the income before contributing. [...]

Answers the question directly. This passage came from another query's row, so it was never in this query's negatives list and `build_filtered_train.py` had nothing to remove (see the README).

## 5. `range for b12` (probably a wrong Yes)

**Labeled positive**

> Normal value ranges for vitamin B12 vary slightly among different laboratories and can range anywhere between 200 to 900 pg/mL. [...]

**Flagged**

> [...] Recommended B12 amounts for adults are 2.4 micrograms per day and recommended amounts for children range from .9 to 1.8 micrograms per day. [...]

The query most likely asks for the normal blood level. This passage gives the recommended daily intake, a different number. The team's final presentation shows this pair as a close call (slide 10). No agreement rate against human judgments was measured, so errors like this one are not counted anywhere.
