# Ethics and data protection checklist (Phase 0, step 0.3) - Data Protection Act 2012 (Act 843)

- [ ] Institutional ethics clearance obtained (record reference number: ______)
- [ ] Ethics committee confirms that analysis of pseudonymised institutional LMS records needs no individual consent, or consent/waiver documented
- [ ] Only pseudonymised IDs (`P00001` style) in any analysis file; no names, index numbers, emails, IP addresses
- [ ] Key file (real_id -> pseudo_id) held by the registrar/lecturer, NOT by the analyst; stored separately and encrypted
- [ ] Retention and deletion date agreed; raw exports kept read-only and outside the shared Git repository
- [ ] Data-sharing statement: no student-level data leaves the institution; only aggregate tables/figures are published
- [ ] Advisor use: alerts are decision support; no automated sanction; students informed that LMS activity is analysed
- [ ] Fairness limits stated (small subgroups are "too small to audit")

`.gitignore` already excludes `oulad/`, `knust_source/`, `data/raw/` and `data/processed/`.
