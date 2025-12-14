# Distilling-Anytime-Trajectories-with-Calibrated-Confidence
Instead of distilling only the best final answer (common in best-of-N / slow-teacher → fast-student), distill the entire improvement trajectory so the student can stream: draft → verified answer → corrected answer → final, each with a well-calibrated confidence and a learnable stop/continue policy.
