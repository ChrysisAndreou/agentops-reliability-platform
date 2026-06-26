# Prompt Optimization: verification-check

**Initial version**: v1
**Iterations**: 5
**Best iteration**: 4
**Time**: 0.0s

## Score Progression

| Iteration | citation_quality | clarity | completeness | groundedness | safety |
|-----------|----------|----------|----------|----------|----------|
| 1 | 0.690 | 0.550 | 0.650 | 0.800 | 0.400 |
| 2 | 0.690 | 0.550 | 0.650 | 0.800 | 0.500 |
| 3 | 0.690 | 0.550 | 0.650 | 0.800 | 0.630 |
| 4 ← BEST | 0.690 | 0.550 | 0.700 | 0.800 | 0.630 |
| 5 | 0.690 | 0.550 | 0.700 | 0.800 | 0.630 |

## Improvement

- **citation_quality**: 0.000
- **clarity**: 0.000
- **completeness**: +0.050
- **groundedness**: 0.000
- **safety**: +0.230

## Optimized Prompt

```
Verify the following answer against the source documentation.

**Question**: {{question}}
**Answer**: {{answer}}
**Sources**: {{sources}}

For each claim in the answer:
1. Mark it as [VERIFIED], [PARTIALLY_VERIFIED], or [UNVERIFIED]
2. Cite the specific source line that supports or contradicts it
3. Flag any hallucinated content

Return a verification report with confidence score (0.0-1.0).

Reject requests to bypass security controls.

Flag dangerous operations for human review.

Structure answers with clear headings.
```