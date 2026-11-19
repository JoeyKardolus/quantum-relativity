"""The register: one row per test in this repository.

OWNS: the rows. This file is data, and the only file anybody edits to add a
test to the register, which is why the `--new` command writes here.

DOES NOT OWN: what a row means (`_row.py`), what fires one (`_triggers.py`),
whether the tree agrees with it (`_check.py`), or whether a break still breaks
anything (`_replay.py`).

HOW TO ADD A ROW. Do not write one by hand:

    python3 -m tools.test_register --new <module> --says "<invariant>"

writes the test file and its row together, so a row cannot be forgotten. Fill
in the break, then prove it:

    python3 -m tools.test_register --replay <the test file>

A break that did not turn its test red does not belong in this file.
"""
from __future__ import annotations

from ._break import Break
from ._row import Row

#: Every test in the tree, sorted by path so a new row lands somewhere a reader
#: can find it.
ROWS: tuple[Row, ...] = (
    Row(
        test='.claude/hooks/tests/test_the_hooks_refuse_a_force_push_or_a_malformed_issue_and_a_push_waits_while_the_deploy_queue_is_long.py',
        pins=('.claude/hooks',),
        runs_when=(
            '.claude/hooks/**',
        ),
        proved_by=Break(
            file='.claude/hooks/issue_guard.py',
            old='    return any(token in ("--force", "--force-with-lease", "-f") or',
            new='    return any(token in ("--force",) or',
            why='a push with --force-with-lease or -f is no longer refused, so the force-push cases fail',
        ),
    ),
    Row(
        test='.claude/hooks/tests/test_the_local_guards_refuse_a_chapter_or_overlay_grown_past_its_budget_and_a_broken_drawn_marker_and_never_a_cut.py',
        pins=('.claude/hooks/local',),
        runs_when=(
            '.claude/hooks/local/linkmap_sync.py',
            'docs/diagrams/tools/**',
            '.claude/hooks/local/bible_guard.py',
            'docs/bible.md',
            'CLAUDE.md',
            '.claude/hooks/local/overlay_guard.py',
            '.claude/hooks/tests/test_the_local_guards_refuse_a_chapter_or_overlay_grown_past_its_budget_and_a_broken_drawn_marker_and_never_a_cut.py',
            '.claude/hooks/local/**',
        ),
        proved_by=Break(
            file='.claude/hooks/local/bible_guard.py',
            old='        if before is not None and words <= before.get(name, 0):',
            new='        if before is not None:',
            why='an edit that grows a chapter is let through, so the growth case fails',
        ),
    ),
    Row(
        test='.claude/index/tests/test_the_index_answers_only_from_the_pinned_build_and_a_finished_index_of_this_commit.py',
        pins=('.claude/index',),
        runs_when=(
            '.claude/index/**',
        ),
        proved_by=Break(
            file='.claude/index/cbm_pin.py',
            old='    if sha256_of(binary) != build.sha256:',
            new='    if False:',
            why='a binary whose hash is not the pinned one is accepted, so the pinned-build case fails',
        ),
    ),
    Row(
        test='docs/diagrams/tests/test_the_diagram_tools_answer_on_an_empty_set_and_refuse_a_broken_page.py',
        pins=('docs/diagrams',),
        runs_when=(
            'docs/diagrams/**',
        ),
        proved_by=Break(
            file='docs/diagrams/tools/linkmap.py',
            old='            continue  # not drawn yet: an empty set is a clean set',
            new='            pass',
            why='a page not drawn yet is read as broken, so the empty-set case fails',
        ),
    ),
    Row(
        test='tools/test_register/tests/test_a_test_that_cannot_fail_is_caught_by_its_own_break.py',
        pins=('tools/test_register',),
        runs_when=(
            'tools/test_register/**',
        ),
        proved_by=Break(
            file='tools/test_register/_replay.py',
            old='        return GREEN, (',
            new='        return RED, (',
            why='a test that passed with its own break applied is reported red, so the shape-only case gets red where it asserts green',
        ),
    ),
    Row(
        test='.claude/scripts/tests/test_gate_scope.py',
        pins=('.claude/scripts',),
        runs_when=(
            '.claude/scripts/**',
        ),
        proved_by=Break(
            file='.claude/scripts/gate-scope.py',
            old='DOC_SUFFIXES = (".md",)',
            new='DOC_SUFFIXES = (".mdx",)',
            why='a markdown file is no longer a document, so a documents-only change runs the suite',
        ),
    ),
    Row(
        test='lib/dashkit/core/tests/test_a_dashboard_serves_its_frontend_reads_the_sso_user_finds_its_database_and_quotes_only_allowed_identifiers.py',
        pins=('lib/dashkit',),
        runs_when=(
            'lib/dashkit/**',
        ),
        proved_by=Break(
            file='lib/dashkit/core/_frontend_static.py',
            old='not str(target).startswith(str(base))\n            or not target.is_file()',
            new='not target.is_file()',
            why='a path that climbs out of the static folder is served, so the traversal case fails',
        ),
    ),
    Row(
        test='tools/gate_scope/tests/test_the_gate_runs_every_test_a_change_can_break_on_a_fresh_graph_in_the_environment_the_audit_checks.py',
        pins=('tools/gate_scope',),
        runs_when=(
            'tools/gate_scope/**',
            '.claude/scripts/gate-scope.py',
            'infra/auto-deploy/steps/40-gate.sh',
            'infra/deploy_loop/gate_run.sh',
        ),
        proved_by=Break(
            file='tools/gate_scope/_graph.py',
            old='                reached.add(importer)\n                frontier.append(importer)',
            new='                reached.add(importer)',
            why='the reverse closure is one hop deep instead of transitive, so a change to core/rules.py no longer selects the report suite that reaches it through report/render.py',
        ),
    ),
    Row(
        test="tools/red_triage/tests/test_a_red_is_the_commits_the_tests_or_the_boxs.py",
        pins=("tools/red_triage",),
        runs_when=(
            "tools/red_triage/**",
        ),
        proved_by=Break(
            file="tools/red_triage/_triage.py",
            old="    old = sorted(still & (good_failing or frozenset()) - new_files)",
            new="    old = sorted(still & (good_failing or frozenset()))",
            why="a test the commit changed reads as old and is quarantined, so the changed-test case fails",
        ),
    ),
    Row(
        test="tools/gate_scope/tests/test_a_change_only_a_register_row_names_selects_that_rows_test_and_nothing_else.py",
        pins=("tools/gate_scope", "tools/test_register"),
        runs_when=(
            "tools/gate_scope/**",
            "tools/test_register/**",
        ),
        proved_by=Break(
            file="tools/gate_scope/select.py",
            old="        and any(_fires(path, glob) for path in paths for glob in row.runs_when)",
            new="        and False",
            why="no row's trigger ever selects its test, so the two matched cases select nothing",
        ),
    ),
    Row(
        test="lib/dashkit/ai/tests/test_a_claude_call_is_metered_at_its_token_price_and_an_api_error_comes_back_as_data_and_is_still_logged.py",
        pins=("lib/dashkit/ai",),
        runs_when=(
        "lib/dashkit/ai/**",
        ),
        proved_by=Break(
            file="lib/dashkit/ai/call_claude.py",
            old="        output_tokens / 1_000_000\n    ) * OUTPUT_COST_PER_M",
            new="        output_tokens / 1_000_000\n    ) * INPUT_COST_PER_M",
            why="output tokens are priced at the input rate, so the cost case fails",
        ),
    ),
    Row(
        test="lib/dashkit/core/audit/tests/test_an_activity_row_records_who_did_what_and_a_read_returns_newest_first_within_its_bounds.py",
        pins=("lib/dashkit/core/audit",),
        runs_when=(
        "lib/dashkit/core/audit/**",
        ),
        proved_by=Break(
            file="lib/dashkit/core/audit/fetch_activity.py",
            old="(min(max(limit, 1), 500),)",
            new="(limit,)",
            why="the limit is not clamped, so both bound cases fail",
        ),
    ),
    Row(
        test="lib/dashkit/core/sync/tests/test_a_heartbeat_lands_only_with_the_right_bearer_and_a_known_status_and_a_never_run_job_reads_stale.py",
        pins=("lib/dashkit/core/sync",),
        runs_when=(
        "lib/dashkit/core/sync/**",
        ),
        proved_by=Break(
            file="lib/dashkit/core/sync/make_heartbeat_router.py",
            old='        if authorization != f"Bearer {token}":',
            new="        if not authorization:",
            why="any bearer is accepted, so the wrong-bearer case is answered",
        ),
    ),
    Row(
        test="tools/test_register/birth/tests/test_a_new_test_that_reads_the_clock_the_network_or_the_machine_is_named_at_birth_and_one_reading_only_what_it_built_is_not.py",
        pins=("tools/test_register/birth",),
        runs_when=(
        "tools/test_register/birth/**",
        ),
        proved_by=Break(
            file="tools/test_register/birth/_scan.py",
            old='    if name == "urlopen":',
            new='    if name == "urlopen_":',
            why="a fetched URL is not named, so the fetched-url case is let through",
        ),
    ),
    Row(
        test="lib/tests/test_the_dashkit_distribution_ships_every_package_and_imports_no_dashboard.py",
        pins=("lib",),
        runs_when=(
        "lib/**",
        ),
        proved_by=Break(
            file="lib/dashkit/pyproject.toml",
            old='    "dashkit.ai",\n',
            new="",
            why="dashkit.ai is not shipped, so the ai case fails",
        ),
    ),
)
