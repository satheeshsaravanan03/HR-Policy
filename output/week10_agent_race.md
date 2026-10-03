# Week 10 — Single-Agent vs Multi-Agent Race

Created: 2026-10-03T18:23:25.602690+00:00

Quality is a deterministic expected-fact/refusal/citation proxy, not an LLM semantic judge. Cost is omitted unless token rates are configured.

| System | Quality | Passed | Mean latency (ms) | Input tokens | Output tokens | Total tokens | Estimated cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-agent MCP baseline | 68.0% | 3/5 | 21976.0 | 2551 | 512 | 3063 | not configured |
| CrewAI specialist team | 16.0% | 1/5 | 7972.9 | 7176 | 3175 | 10351 | not configured |

## Provisional verdict

The single-agent baseline is the provisional keeper: the team does not improve the deterministic quality proxy (delta -52.0 percentage points), while mean latency changes by -14003.1 ms and total tokens by +7288. Keep multi-agent delegation for cases where specialist separation or parallel work is specifically useful.


## Per-case results

| Case | Single quality | Team quality | Single latency ms | Team latency ms | Team specialists |
|---|---:|---:|---:|---:|---|
| W10-POL-01 | 100% | 80% | 23680 | 39864 | policy_research, evidence_review |
| W10-POL-02 | 20% | 0% | 22712 | 0 |  |
| W10-POL-03 | 20% | 0% | 22732 | 0 |  |
| W10-EMP-01 | 100% | 0% | 15358 | 0 |  |
| W10-REF-01 | 100% | 0% | 25398 | 0 |  |

### W10-POL-01 — How many annual leave days does an Acme full-time employee receive per calendar year?

**Single-agent:**

Acme full‑time employees are entitled to **18 days of annual leave per calendar year**. [CITE: ACME-LEAVE-2026#structure-0002-836f5471]

**CrewAI team:**

**Policy ID:** ACME-LEAVE-2026  
**Section:** 2  
**Chunk ID:** ACME-LEAVE-2026#structure-0002-836f5471  

```
## 2. Annual leave allowance  
Employees receive 18 days of annual leave per calendar year. Employees joining after 1 July receive a prorated allowance based on completed months of service.
```

**CrewAI specialist tasks/status:**

```json
[
  {
    "role": "policy_research",
    "state": "completed",
    "tool_calls": [
      "search_hr_policy"
    ]
  }
]
```

### W10-POL-02 — How many unused Acme annual-leave days may be carried into the next year?

**Single-agent:**

Employees can carry forward **up to 10 unused annual‑leave days** into the next calendar year. [CITE: ACME-LEAVE-2026#structure-0003-6a055a77]

**CrewAI team:**

Run failed: RateLimitError

**CrewAI specialist tasks/status:**

```json
[]
```

### W10-POL-03 — For non-technical staff, how many unused leaves can be carried over to the next year?

**Single-agent:**

For non‑technical staff, up to **nine** unused leaves may be carried over to the next year. [CITE: SS-HB-2025#structure-0066-0a3777d2]

**CrewAI team:**

Run failed: RateLimitError

**CrewAI specialist tasks/status:**

```json
[]
```

### W10-EMP-01 — Show the policy-backed leave summary for EMP-002, including current balance, carry-forward and compensatory leave.

**Single-agent:**

Employee `EMP-002` leave summary:
- Current leave balance: **14 days**
- Annual leave entitlement: **24 days**
- Carry-forward available: **10 days**
- Compensatory leave available: **3 days**

Policy: `AZURE-HR-2026`

**CrewAI team:**

Run failed: RateLimitError

**CrewAI specialist tasks/status:**

```json
[]
```

### W10-REF-01 — What is SoftSuave's sabbatical leave entitlement?

**Single-agent:**

I cannot answer this from the indexed policy documents.
Reason: no document contains sabbatical.

**CrewAI team:**

Run failed: RateLimitError

**CrewAI specialist tasks/status:**

```json
[]
```
