- The report produced is not acceptable:
the result it gives is vague and not structured and quantitative at all.
the following metrics should be considered and quantified:
- task completion: did it do what it was supposed to do?
    task_completion_rate = successful_positive_scenarios / total_positive_scenarios
- boundary adherence: does it correctly refuse what it should refuse?
    boundary_adherence_rate = successful_negative_scenarios / total_negative_scenarios
- state integrity: when it says it did something, did it actually do it?
    state_integrity_rate = confirmed_by_probe / total_probe_scenarios
    (here the name of metric can be enhanced. Users might not know what is state.)
- Constraint violation breakdown: which rules does it break the most?
    example:
    per_constraint_violation_rate:
    "only verified entities can book": vioalted 4/6 times (67%)
    "slots must be within 09:00 - 18:00": violated 1/2 times (50%)
- recovery behavior: how does it handle adversarial inputs?
    recovery_rate = handled_gracefully / total_adversarial_scenarios
    Negative scenarios where the agent fails aren't all equal. An agent that says "I'm sorry, I can't help with that" is better than one that crashes, loops, or leaks internal information. The scorer can classify the failure type — clean refusal vs confused response vs error.
- conversation efficiency: did it accomplish the task in a reasonable number of turns?
    example:
    avg_turns_to_completion: 3.2 turns (positive scenarios that passed)
    avg_turns_to_refusal: 1.8 turns (negative scenarios that passed)
