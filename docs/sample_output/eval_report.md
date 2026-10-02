# RAG Evaluation Report

Generated: 2026-10-02T18:18:15+00:00  
Configuration: embedding=local, generator=extractive, top_k=5, rerank=True, refusal_threshold=0.12  
Questions: 15 in-scope, 2 out-of-scope

## Metrics

| Metric | Value |
|---|---|
| hit@1 | 0.933 |
| hit@3 | 1.000 |
| hit@5 | 1.000 |
| mrr | 0.967 |
| citation_coverage | 1.000 |
| citation_accuracy | 1.000 |
| false_refusal_rate | 0.000 |
| oos_refusal_accuracy | 1.000 |

## Per-question results

| ID | Expected doc | Top doc | Rank | Max sim | Refused | Correct refusal decision |
|---|---|---|---|---|---|---|
| q01 | ho-property-coverages | ho-property-coverages | 1 | 0.339 | False | True |
| q02 | ho-property-coverages | ho-property-coverages | 1 | 0.191 | False | True |
| q03 | ho-perils-exclusions | claims-water-damage-sop | 2 | 0.177 | False | True |
| q04 | ho-perils-exclusions | ho-perils-exclusions | 1 | 0.243 | False | True |
| q05 | ho-conditions | ho-conditions | 1 | 0.52 | False | True |
| q06 | ho-conditions | ho-conditions | 1 | 0.151 | False | True |
| q07 | ho-liability | ho-liability | 1 | 0.224 | False | True |
| q08 | uw-homeowners-guideline | uw-homeowners-guideline | 1 | 0.446 | False | True |
| q09 | uw-homeowners-guideline | uw-homeowners-guideline | 1 | 0.286 | False | True |
| q10 | uw-referral-authority | uw-referral-authority | 1 | 0.2 | False | True |
| q11 | claims-fnol-sop | claims-fnol-sop | 1 | 0.401 | False | True |
| q12 | claims-fnol-sop | claims-fnol-sop | 1 | 0.327 | False | True |
| q13 | claims-water-damage-sop | claims-water-damage-sop | 1 | 0.3 | False | True |
| q14 | fraud-red-flags | fraud-red-flags | 1 | 0.382 | False | True |
| q15 | ho-definitions | ho-definitions | 1 | 0.392 | False | True |
| q16 | (out of scope) | ho-perils-exclusions | - | 0.046 | True | True |
| q17 | (out of scope) | ho-property-coverages | - | 0.055 | True | True |

## Sample answers

**q01: What is the Coverage B limit for detached structures like a fence or shed?**

> We cover other structures on the residence premises that are set apart from the dwelling by clear space, such as a detached garage, fence or garden shed. [ho-property-coverages#coverage-b-other-structures-0] The Coverage B limit is 10 percent of the Coverage A limit unless a higher limit is shown on the declarations. [ho-property-coverages#coverage-b-other-structures-0] Property usually located at a secondary residence is limited to 10 percent of the Coverage C limit. [ho-property-coverages#coverage-c-personal-property-0]

**q02: How much does the policy pay for theft of jewelry and watches?**

> The following special limits apply per loss: 200 dollars for cash and coins; 1,500 dollars for securities and passports; 1,500 dollars for watercraft including trailers; 2,500 dollars for theft of jewelry, watches and furs; 2,500 dollars for theft of firearms; and 2,500 dollars for business property on the residence premises. [ho-property-coverages#special-limits-of-liability-0] The insured recently increased Coverage C limits or added scheduled jewelry shortly before a theft. [fraud-red-flags#policy-red-flags-0]

**q03: Is sewer or drain backup covered, or is it excluded with flood?**

> First establish whether the water was a sudden and accidental discharge, which is covered, or flood, surface water, sewer backup or long-term seepage, which are excluded unless an endorsement applies. [claims-water-damage-sop#coverage-determination-0] Water that backs up through sewers or drains is also excluded unless the Water Backup endorsement NM-END-14 is attached. [ho-perils-exclusions#exclusion-flood-0]

**q04: Does the policy cover earthquake or sinkhole damage?**

> We do not cover earthquake, landslide, mudflow, sinkhole or other earth sinking, rising or shifting. [ho-perils-exclusions#exclusion-earth-movement-0] If earth movement results in a fire or explosion, we do cover the resulting fire or explosion damage. [ho-perils-exclusions#exclusion-earth-movement-0] "Occurrence" means an accident, including continuous or repeated exposure to substantially the same harmful conditions, that results in bodily injury or property damage during the policy period. [ho-definitions#occurrence-0]

**q16: What is the best recipe for sourdough bread starter?**

> I don't know based on the indexed documents.

**q17: Who won the football world cup in 1998?**

> I don't know based on the indexed documents.
