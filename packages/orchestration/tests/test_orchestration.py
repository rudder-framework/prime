"""Tests for the orchestration package."""
import pytest

class TestPipeline:
    def test_stage_count(self):
        from orchestration.pipeline import STAGES
        assert len(STAGES) == 16

    def test_no_circular_deps(self):
        from orchestration.pipeline import STAGES
        stage_names = {s.name for s in STAGES}
        for s in STAGES:
            for dep in s.depends_on:
                assert dep in stage_names, f'{s.name} depends on unknown stage {dep}'

    def test_execution_order(self):
        from orchestration.pipeline import Pipeline
        p = Pipeline()
        order = p.get_execution_order()
        assert len(order) == 16
        names = [s.name for s in order]
        # typology must come before vector
        assert names.index('typology') < names.index('vector')
        # vector before eigendecomp
        assert names.index('vector') < names.index('eigendecomp')

    def test_subset_includes_deps(self):
        from orchestration.pipeline import Pipeline
        p = Pipeline()
        order = p.get_execution_order(include=['geometry'])
        names = [s.name for s in order]
        assert 'typology' in names  # transitive dep
        assert 'vector' in names
        assert 'eigendecomp' in names
        assert 'geometry' in names

    def test_skip_optional(self):
        from orchestration.pipeline import Pipeline
        p = Pipeline()
        order = p.get_execution_order(skip_optional=True)
        names = [s.name for s in order]
        assert 'divergence' not in names  # optional
        assert 'topology' not in names  # optional

    def test_list_stages(self):
        from orchestration.pipeline import Pipeline
        p = Pipeline()
        stages = p.list_stages()
        assert len(stages) == 16
        assert all('name' in s for s in stages)

    def test_plan_dry_run(self):
        from orchestration.pipeline import Pipeline
        p = Pipeline()
        plan = p.plan()
        assert len(plan) == 16
        # All stages return a status (dry_run or skipped due to missing inputs)
        assert all('status' in r for r in plan)
