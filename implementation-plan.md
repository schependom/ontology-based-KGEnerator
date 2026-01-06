# Implementation Plan: KGenerator Upgrade

This plan details the modifications required to upgrade the Knowledge Graph Generator with **Structural Signature Sampling**, **Instance Looping**, and **Memory Optimizations**.

## Goals
1.  **Diversity:** Implement "Signature Sampling" to ensure diverse reasoning paths are included.
2.  **Volume:** Implement "Instance Looping" to generate multiple distinct facts per rule.
3.  **Efficiency:** Optimize recursion with `itertools` limiting and "Inverse Loop" pruning.
4.  **Flexibility:** Make the new sampling logic optional via flags.

---

## 1. Modify `chainer.py`

### A. Update `__init__`
Add a flag to enable/disable signature sampling.

- **File:** `chainer.py`
- **Class:** `BackwardChainer`
- **Method:** `__init__`
- **Action:** Add argument `use_signature_sampling: bool = True` and store it in `self.use_signature_sampling`.

### B. Add Helper Method `_get_proof_signature`
Logic to extract the structure of a proof.

- **File:** `chainer.py`
- **Class:** `BackwardChainer`
- **Action:** Add the following method:

```python
    def _get_proof_signature(self, proof: Proof) -> Tuple[str, ...]:
        """
        Generates a signature based on the sequence of rules used.
        Returns a sorted tuple of rule names to identify structural identity.
        """
        rules = []
        if proof.rule:
            rules.append(proof.rule.name)
        for sub_proof in proof.sub_proofs:
            rules.extend(self._get_proof_signature(sub_proof))
        return tuple(sorted(rules))
```

### C. Update _find_proofs_recursive (The Core Logic)

Refactor the loop generation to handle memory limits, inverse pruning, and signature sampling.

- **File:** `chainer.py`
- **Class:** `BackwardChainer`
- **Method:** `_find_proofs_recursive`

Action: Replace the generation logic loop with this enhanced version:

Key Changes to Insert:

Inverse Loop Pruning: Before processing a rule, check if rule.conclusion.predicate is the inverse of parent_predicate. If so, continue.

Memory Safety: When generating sub_proof_combination via itertools.product, do not cast to list immediately. Iterate and break if count > 100 (or a defined constant) to prevent RAM explosion.

Signature Sampling (Conditional):

If self.use_signature_sampling is True: Collect valid proofs into a list, group them by _get_proof_signature, and yield one random proof per group.

If False: Yield proofs immediately as they are found (standard behavior).

Debug Prints: Add if self.verbose: blocks to print how many signatures were found vs how many proofs total.

## 2. Modify generate.py

### 2A. Update generate_proofs_for_rule

Implement "Instance Looping" to create volume (many distinct facts).

- **File:** `generate.py`
- **Class:** `KGenerator`
- **Method:** `generate_proofs_for_rule`

Class: KGenerator

Method: generate_proofs_for_rule

Action:

Add arguments: n_instances: int = 1 and proofs_per_instance: int = 1.

Wrap the generation logic in a loop: for _ in range(n_instances):.

Inside the loop, call self.chainer.generate_proof_trees(rule_name).

Collect proofs_per_instance proofs from that generator and add them to the main list.

Return the combined list of proofs (e.g., 20 instances * 2 proofs = 40 proofs total).

## 3. Modify create_data.py

### 3A. Update __init__

Pass configuration flags down to the generator.

- **File:** `create_data.py`
- **Class:** `KGEDatasetGenerator`
- **Method:** `__init__`

File: create_data.py

Class: KGEDatasetGenerator

Method: __init__

Action: Add arguments use_signature_sampling: bool = True. Pass this to the KGenerator initialization (requires updating KGenerator init in generate.py as well to accept/pass this flag to BackwardChainer).

### 3B. Update _generate_one_sample

Use the new Instance Looping parameters.

- **File:** `create_data.py`
- **Class:** `KGEDatasetGenerator`
- **Method:** `_generate_one_sample`

Action:

Define n_instances_per_rule = random.randint(10, 20) (or a configurable parameter).

Update the call to self.generator.generate_proofs_for_rule:

Python
proofs = self.generator.generate_proofs_for_rule(
    rule.name,
    n_instances=n_instances_per_rule,
    proofs_per_instance=min_proofs_per_rule
)
Remove the old logic that sliced [:10000] since volume is now controlled by n_instances.

## 4. Verification Checklist

After applying changes, verify:

Volume: Does a generated sample contain ~10-20 distinct "Fathers" (or target relations) instead of just 1?

Diversity: When use_signature_sampling=True, do the proofs for a single relation (e.g., Cousin) show different reasoning paths (Father-side vs Mother-side)?

Memory: Does generation speed stay stable even with deep recursion (due to itertools cap)?

Inverse Loops: Are infinite loops like Child->Parent->Child reduced/eliminated in the logs?