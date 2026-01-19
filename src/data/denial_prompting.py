"""
Denial Prompting Utilities

Handles curriculum learning and denial prompt augmentation for RL training.
"""

from typing import List, Dict, Any, Optional


class CurriculumScheduler:
    """
    Manages curriculum learning schedule for denial prompting.

    The curriculum gradually increases the number of denial constraints
    during training to avoid sparse reward problems early on.
    """

    def __init__(self, schedule: List[Dict[str, Any]]):
        """
        Initialize curriculum scheduler.

        Args:
            schedule: List of curriculum stages, each with:
                - stage: int (stage number)
                - steps: [start, end] (step range)
                - num_constraints: int (number of constraints to deny)
                - description: str (optional description)

        Example:
            schedule = [
                {'stage': 1, 'steps': [0, 1000], 'num_constraints': 0},
                {'stage': 2, 'steps': [1001, 2500], 'num_constraints': 1},
                {'stage': 3, 'steps': [2501, 5000], 'num_constraints': 2},
            ]
        """
        self.schedule = sorted(schedule, key=lambda x: x['steps'][0])
        self._validate_schedule()

    def _validate_schedule(self):
        """Validate that schedule is well-formed."""
        if not self.schedule:
            raise ValueError("Schedule cannot be empty")

        # Check for gaps or overlaps
        for i in range(len(self.schedule) - 1):
            current_end = self.schedule[i]['steps'][1]
            next_start = self.schedule[i + 1]['steps'][0]

            if current_end >= next_start:
                raise ValueError(
                    f"Schedule overlap or gap between stage {i} and {i+1}: "
                    f"stage {i} ends at {current_end}, "
                    f"stage {i+1} starts at {next_start}"
                )

    def get_num_constraints(self, step: int) -> int:
        """
        Get number of constraints for current training step.

        Args:
            step: Current training step

        Returns:
            Number of constraints to use
        """
        for stage in self.schedule:
            start_step, end_step = stage['steps']
            if start_step <= step <= end_step:
                return stage['num_constraints']

        # If beyond all stages, use last stage's constraints
        return self.schedule[-1]['num_constraints']

    def get_stage(self, step: int) -> Optional[Dict[str, Any]]:
        """
        Get the current curriculum stage.

        Args:
            step: Current training step

        Returns:
            Stage dictionary or None if step is out of range
        """
        for stage in self.schedule:
            start_step, end_step = stage['steps']
            if start_step <= step <= end_step:
                return stage

        # Return last stage if beyond schedule
        if step > self.schedule[-1]['steps'][1]:
            return self.schedule[-1]

        return None

    def get_progress(self, step: int) -> float:
        """
        Get curriculum progress (0.0 to 1.0).

        Args:
            step: Current training step

        Returns:
            Progress through curriculum (0.0 = start, 1.0 = end)
        """
        total_steps = self.schedule[-1]['steps'][1]
        return min(1.0, step / total_steps)

    def summary(self) -> str:
        """Get a human-readable summary of the curriculum."""
        lines = ["Curriculum Schedule:"]
        lines.append("=" * 60)

        for stage in self.schedule:
            start, end = stage['steps']
            num_constraints = stage['num_constraints']
            desc = stage.get('description', '')

            lines.append(
                f"Stage {stage['stage']}: Steps {start}-{end} "
                f"→ {num_constraints} constraints"
            )
            if desc:
                lines.append(f"  ({desc})")

        return "\n".join(lines)


class DenialPromptAugmenter:
    """
    Augments prompts with denial constraints.

    Takes a base prompt and adds denial instructions based on
    the NeoCoder denial prompting methodology.
    """

    def __init__(self, template: Optional[str] = None):
        """
        Initialize augmenter.

        Args:
            template: Optional custom template for denial instructions.
                     Use {constraints} as placeholder for the constraint list.
        """
        self.template = template or self._default_template()

    @staticmethod
    def _default_template() -> str:
        """Default template matching NeoCoder format."""
        return (
            "{original_prompt}\n\n"
            "Programming constraints: DO NOT use the following techniques\n"
            "{constraints}"
        )

    def augment(
        self,
        original_prompt: str,
        constraints: List[str],
        format_style: str = "list"
    ) -> str:
        """
        Augment a prompt with denial constraints.

        Args:
            original_prompt: The original problem statement
            constraints: List of techniques to deny (e.g., ["for loop", "list comprehension"])
            format_style: How to format constraints:
                - "list": Bullet point list (default)
                - "inline": Comma-separated inline
                - "numbered": Numbered list

        Returns:
            Augmented prompt with denial instructions

        Example:
            >>> augmenter = DenialPromptAugmenter()
            >>> prompt = "Write a function to sum a list"
            >>> constraints = ["for loop", "sum() builtin"]
            >>> augmented = augmenter.augment(prompt, constraints)
            >>> print(augmented)
            Write a function to sum a list

            Programming constraints: DO NOT use the following techniques
            - for loop
            - sum() builtin
        """
        if not constraints:
            return original_prompt

        # Format constraints based on style
        if format_style == "list":
            formatted_constraints = "\n".join(f"- {c}" for c in constraints)
        elif format_style == "inline":
            formatted_constraints = ", ".join(constraints)
        elif format_style == "numbered":
            formatted_constraints = "\n".join(
                f"{i+1}. {c}" for i, c in enumerate(constraints)
            )
        else:
            raise ValueError(f"Unknown format_style: {format_style}")

        # Apply template
        return self.template.format(
            original_prompt=original_prompt,
            constraints=formatted_constraints
        )

    def batch_augment(
        self,
        prompts: List[str],
        constraints_list: List[List[str]],
        **kwargs
    ) -> List[str]:
        """
        Augment multiple prompts at once.

        Args:
            prompts: List of original prompts
            constraints_list: List of constraint lists (one per prompt)
            **kwargs: Additional arguments passed to augment()

        Returns:
            List of augmented prompts
        """
        return [
            self.augment(prompt, constraints, **kwargs)
            for prompt, constraints in zip(prompts, constraints_list)
        ]


if __name__ == "__main__":
    # Test curriculum scheduler
    print("="*80)
    print("Testing Curriculum Scheduler")
    print("="*80)

    curriculum = [
        {'stage': 1, 'steps': [0, 100], 'num_constraints': 0, 'description': 'Baseline'},
        {'stage': 2, 'steps': [101, 200], 'num_constraints': 1, 'description': 'Easy denial'},
        {'stage': 3, 'steps': [201, 300], 'num_constraints': 2, 'description': 'Medium denial'},
    ]

    scheduler = CurriculumScheduler(curriculum)
    print(scheduler.summary())

    print("\nTesting at different steps:")
    for step in [50, 150, 250, 350]:
        num_constraints = scheduler.get_num_constraints(step)
        progress = scheduler.get_progress(step)
        stage = scheduler.get_stage(step)
        print(f"  Step {step}: {num_constraints} constraints (progress: {progress:.1%}, stage: {stage['stage'] if stage else 'N/A'})")

    # Test denial prompt augmenter
    print("\n" + "="*80)
    print("Testing Denial Prompt Augmenter")
    print("="*80)

    augmenter = DenialPromptAugmenter()

    original = "Write a function to sort a list of integers in ascending order."
    constraints = ["for loop", "sorted() builtin", "list.sort() method"]

    print("\nOriginal prompt:")
    print(original)

    print("\n--- List format ---")
    augmented = augmenter.augment(original, constraints, format_style="list")
    print(augmented)

    print("\n--- Inline format ---")
    augmented = augmenter.augment(original, constraints, format_style="inline")
    print(augmented)

    print("\n--- Numbered format ---")
    augmented = augmenter.augment(original, constraints, format_style="numbered")
    print(augmented)

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
