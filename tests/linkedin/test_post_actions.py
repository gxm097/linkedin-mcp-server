"""Tests for the comment-action owner.

Two contracts carry most of the weight here.

The first is ``retry_safe``. A comment that posted and failed to confirm is
indistinguishable on the page from one that never left the box, so the flag
has to go false at the click and stay false through both endings. A caller
that keys a retry on ``posted`` alone comments twice, which is the failure
these cases exist to catch.

The second is that the dry run is a read. ``confirm_comment`` False may probe
the surface and may not type, so those cases assert the write program never
ran rather than only reading the status back.

``page.evaluate`` is a mock, so none of the extractor JS executes: the browser
answers are supplied directly and what is under test is the state machine that
reads them. The scripts are dispatched by identity against the module's own
constants, so a renamed program fails loudly instead of silently matching
nothing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin_mcp_server.core.exceptions import InvalidReferenceError
from linkedin_mcp_server.linkedin import post_actions as module
from linkedin_mcp_server.linkedin.identifiers import normalize_post_reference
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.post_actions import PostActions
from linkedin_mcp_server.linkedin.session import PageSession


ACTIVITY_URN = "urn:li:activity:7123456789012345678"
ACTIVITY_PATH = f"/feed/update/{ACTIVITY_URN}/"
ACTIVITY_URL = f"https://www.linkedin.com{ACTIVITY_PATH}"
SLUG = "jane-doe-ab12cd-activity-7123456789012345678-xY7z"


class _Handle:
    """A stand-in for the pinned owner handle, scripted per program."""

    def __init__(self, answers: dict[str, Any]):
        self._answers = answers
        self.ran: list[str] = []

    def as_element(self) -> object | None:
        return self

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        name = _PROGRAMS.get(script)
        assert name is not None, "an unrecognized program reached the owner handle"
        self.ran.append(name)
        answer = self._answers.get(name, _MISSING)
        assert answer is not _MISSING, f"this case did not script {name!r}"
        return answer() if callable(answer) else answer

    async def dispose(self) -> None:
        self.ran.append("dispose_handle")


_MISSING = object()

#: Program identity -> the name a case scripts it under.
_PROGRAMS = {
    module._COMMENT_WRITE_JS: "write",
    module._COMMENT_SUBMIT_READY_JS: "submit_ready",
    module._COMMENT_SUBMIT_JS: "submit",
    module._COMMENT_CONFIRM_JS: "confirm",
    module._COMMENT_CLEANUP_JS: "cleanup",
    module._COMMENT_DISPOSE_JS: "dispose",
}


def _actions(
    monkeypatch: pytest.MonkeyPatch,
    *,
    surface: str,
    excerpt: Any = None,
    owner: _Handle | None = None,
) -> tuple[PostActions, MagicMock]:
    """Wire the comment owner with a browser that answers as scripted.

    *surface* is what the comment-surface probe reports. *excerpt* is what the
    post-text read answers, defaulting to the page that could not be read.
    *owner* is the pinned handle, or ``None`` for a page the editor could not
    be pinned on.
    """
    page = MagicMock()
    page.url = ACTIVITY_URL

    # run_on_linkedin wraps each program in a landing check, so the two page
    # programs are told apart by containment where the owner's are told apart
    # by identity. A bare return value is the documented test-double path,
    # where the driver's address stands in for the document's.
    async def evaluate(script: str, arg: Any = None) -> Any:
        if module._COMMENT_EXCERPT_JS in script:
            return excerpt
        assert module._COMMENT_SURFACE_JS in script, (
            "an unrecognized program reached the page"
        )
        return surface

    page.evaluate = AsyncMock(side_effect=evaluate)
    page.evaluate_handle = AsyncMock(return_value=owner if owner else _unpinnable())

    monkeypatch.setattr(PageSession, "check_rate_limit", AsyncMock())
    monkeypatch.setattr(PageNavigator, "_navigate_to_page", AsyncMock())

    session = PageSession(page)
    return PostActions(session, PageNavigator(session)), page


def _unpinnable() -> MagicMock:
    """A handle for a page with no comment box: it holds no element."""
    handle = MagicMock()
    handle.as_element = MagicMock(return_value=None)
    handle.evaluate = AsyncMock(return_value=True)
    handle.dispose = AsyncMock()
    return handle


class TestPostReference:
    """What counts as a reference to a post, with no browser involved."""

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            (ACTIVITY_URN, ACTIVITY_PATH),
            (f"urn:li:ugcPost:{'7' * 19}", f"/feed/update/urn:li:ugcPost:{'7' * 19}/"),
            (f"urn:li:share:{'7' * 19}", f"/feed/update/urn:li:share:{'7' * 19}/"),
            (ACTIVITY_URL, ACTIVITY_PATH),
            (ACTIVITY_PATH, ACTIVITY_PATH),
            (f"linkedin.com{ACTIVITY_PATH}", ACTIVITY_PATH),
            (f"https://www.linkedin.com{ACTIVITY_PATH}?utm_source=x", ACTIVITY_PATH),
            (SLUG, f"/posts/{SLUG}"),
            (f"https://www.linkedin.com/posts/{SLUG}", f"/posts/{SLUG}"),
            (f"https://de.linkedin.com/posts/{SLUG}", f"/posts/{SLUG}"),
        ],
    )
    def test_accepts_every_form_a_read_tool_hands_back(
        self, given: str, expected: str
    ) -> None:
        assert normalize_post_reference(given) == expected

    @pytest.mark.parametrize(
        "given",
        [
            "",
            "   ",
            "https://www.linkedin.com/in/jane-doe/",
            "https://www.linkedin.com/company/acme/",
            # A comment is not a post, and neither is a reaction.
            "urn:li:comment:7123456789012345678",
            "urn:li:reaction:7123456789012345678",
            # Under the id floor: a fixture-shaped number, not a live entity.
            "urn:li:activity:123",
            # A dot segment retargets the path a browser would load.
            f"https://www.linkedin.com/feed/update/../../in/bob/{ACTIVITY_URN}/",
            "https://example.com/feed/update/urn:li:activity:7123456789012345678/",
            "not a reference at all",
        ],
    )
    def test_refuses_what_names_no_post(self, given: str) -> None:
        with pytest.raises(InvalidReferenceError):
            normalize_post_reference(given)

    def test_refuses_a_short_link_only_a_redirect_resolves(self) -> None:
        with pytest.raises(InvalidReferenceError, match="shortened"):
            normalize_post_reference("https://lnkd.in/abc123")


class TestCommentTextRefusals:
    """Text judged before a browser session is acquired."""

    @pytest.mark.parametrize("comment", ["", "   ", "\n\n", "ok\tthen", "a\x00b"])
    async def test_unsafe_text_never_reaches_the_page(
        self, monkeypatch: pytest.MonkeyPatch, comment: str
    ) -> None:
        actions, page = _actions(monkeypatch, surface="ready")

        result = await actions.comment_on_post(
            ACTIVITY_URN, comment, confirm_comment=True
        )

        assert result["status"] == "invalid_comment"
        assert result["posted"] is False
        assert result["retry_safe"] is True
        page.evaluate.assert_not_awaited()
        page.evaluate_handle.assert_not_awaited()


class TestSurfaceRefusals:
    """What the probe reports, and the status each report produces."""

    @pytest.mark.parametrize(
        ("surface", "status"),
        [
            ("off_route", "post_unavailable"),
            ("absent", "comments_unavailable"),
            ("ambiguous", "comments_unavailable"),
            # A box that cannot be tied to a post card could be typed into
            # while sitting under a different post, so it is refused rather
            # than reported as ready.
            ("no_card", "comments_unavailable"),
            ("occupied", "editor_occupied"),
        ],
    )
    async def test_a_refused_surface_is_never_typed_into(
        self, monkeypatch: pytest.MonkeyPatch, surface: str, status: str
    ) -> None:
        monkeypatch.setattr(module, "_COMMENT_SURFACE_TIMEOUT_SECONDS", 0.0)
        actions, page = _actions(monkeypatch, surface=surface)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == status
        assert result["posted"] is False
        # Nothing was submitted, so a retry is the caller's to make freely.
        assert result["retry_safe"] is True
        page.evaluate_handle.assert_not_awaited()

    async def test_an_unpinnable_box_stops_before_the_editor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actions, _ = _actions(monkeypatch, surface="ready", owner=None)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == "comments_unavailable"
        assert result["retry_safe"] is True


class TestDryRun:
    """``confirm_comment`` False is a read."""

    async def test_it_reports_the_surface_and_types_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        owner = _Handle({})
        actions, page = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=False
        )

        assert result["status"] == "dry_run"
        assert result["posted"] is False
        assert result["retry_safe"] is True
        assert result["url"] == ACTIVITY_URL
        # The editor was never even pinned, let alone written to.
        page.evaluate_handle.assert_not_awaited()
        assert owner.ran == []

    async def test_a_dry_run_refuses_an_occupied_box_without_typing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        owner = _Handle({})
        actions, _ = _actions(monkeypatch, surface="occupied", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=False
        )

        assert result["status"] == "editor_occupied"
        assert owner.ran == []


class TestDryRunExcerpt:
    """The dry run says *which* post it reached, not only that it accepts one.

    A permalink out of ``search_posts`` is not aligned to the result it sat
    beside, so a caller that means a particular post cannot tell from the rest
    of the result whether it landed there. These cases are about the excerpt
    being trustworthy where it is present and absent where it is not: a caller
    that matches on text it was handed will comment on whatever that text came
    from, so a wrong excerpt is worse than none at all.
    """

    async def test_the_post_text_comes_back_with_the_dry_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actions, _ = _actions(
            monkeypatch,
            surface="ready",
            excerpt={"text": "Synerva shipped a thing today.", "truncated": False},
        )

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=False
        )

        assert result["status"] == "dry_run"
        assert result["excerpt"] == "Synerva shipped a thing today."
        assert result["excerpt_truncated"] is False

    async def test_a_cut_short_excerpt_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without the flag, a phrase missing from the excerpt reads as a phrase
        # missing from the post, and a mention check answers no on a post that
        # carries the mention further down.
        actions, _ = _actions(
            monkeypatch,
            surface="ready",
            excerpt={"text": "A long post about", "truncated": True},
        )

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=False
        )

        assert result["excerpt"] == "A long post about"
        assert result["excerpt_truncated"] is True

    @pytest.mark.parametrize(
        "answer",
        [
            None,
            # Off the route, the program returns nothing rather than text from
            # whatever page the browser is actually sitting on.
            {"text": "", "truncated": False},
            {"text": None, "truncated": False},
            {"truncated": False},
            # A shape the program does not produce is not trusted either.
            "ready",
            [],
        ],
        ids=[
            "no answer",
            "empty text",
            "null text",
            "no text key",
            "a bare string",
            "a list",
        ],
    )
    async def test_text_that_could_not_be_read_is_null_not_guessed(
        self, monkeypatch: pytest.MonkeyPatch, answer: Any
    ) -> None:
        actions, _ = _actions(monkeypatch, surface="ready", excerpt=answer)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=False
        )

        assert result["status"] == "dry_run"
        assert result["excerpt"] is None
        assert result["excerpt_truncated"] is False

    async def test_a_failed_read_does_not_fail_the_dry_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Losing the excerpt is worth less than losing the surface verdict."""
        actions, page = _actions(monkeypatch, surface="ready")
        surface = page.evaluate.side_effect

        async def evaluate(script: str, arg: Any = None) -> Any:
            if module._COMMENT_EXCERPT_JS in script:
                raise RuntimeError("the page went away mid-read")
            return await surface(script, arg)

        page.evaluate = AsyncMock(side_effect=evaluate)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=False
        )

        assert result["status"] == "dry_run"
        assert result["excerpt"] is None

    @pytest.mark.parametrize(
        "surface",
        ["off_route", "absent", "ambiguous", "no_card", "occupied"],
    )
    async def test_a_surface_that_refuses_carries_no_excerpt_key(
        self, monkeypatch: pytest.MonkeyPatch, surface: str
    ) -> None:
        # The excerpt is read after the surface clears, so a refusal answers
        # without the field rather than with a null that reads as "unreadable".
        monkeypatch.setattr(module, "_COMMENT_SURFACE_TIMEOUT_SECONDS", 0.0)
        actions, _ = _actions(
            monkeypatch,
            surface=surface,
            excerpt={"text": "should never be read", "truncated": False},
        )

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=False
        )

        assert result["status"] != "dry_run"
        assert "excerpt" not in result
        assert "excerpt_truncated" not in result

    async def test_a_confirmed_comment_does_not_read_the_post(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The write path is guarded by the route, not by text it read back."""
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": "ready",
                "submit": "clicked",
                "confirm": "confirmed",
                "dispose": True,
            }
        )
        actions, page = _actions(
            monkeypatch,
            surface="ready",
            excerpt={"text": "should never be read", "truncated": False},
            owner=owner,
        )

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == "posted"
        assert "excerpt" not in result
        ran = [call.args[0] for call in page.evaluate.await_args_list]
        assert not any(module._COMMENT_EXCERPT_JS in script for script in ran)


class TestWriteFailures:
    """Every ending that leaves the box without submitting."""

    @pytest.mark.parametrize(
        ("written", "status"),
        [
            ("occupied", "editor_occupied"),
            ("unsupported", "editor_unsupported"),
            ("invalid", "editor_changed"),
        ],
    )
    async def test_a_failed_write_submits_nothing_and_stays_retry_safe(
        self, monkeypatch: pytest.MonkeyPatch, written: str, status: str
    ) -> None:
        owner = _Handle({"write": written, "cleanup": True, "dispose": True})
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == status
        assert result["posted"] is False
        assert result["retry_safe"] is True
        assert "submit" not in owner.ran

    async def test_a_button_that_never_enables_clears_the_text_it_wrote(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(module, "_COMMENT_SUBMIT_READY_TIMEOUT_SECONDS", 0.0)
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": "disabled",
                "cleanup": True,
                "dispose": True,
            }
        )
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == "submit_not_ready"
        assert result["retry_safe"] is True
        # The comment this call typed does not outlive it.
        assert "cleanup" in owner.ran
        assert "submit" not in owner.ran

    async def test_a_control_that_never_appears_clears_the_text_it_wrote(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``absent`` is the post page's own first answer, not a failure.

        The control that posts a comment is created from the editor's input
        event, so it does not exist at the moment the text lands. A run that
        treated the first ``absent`` as fatal would never post anything.
        """
        monkeypatch.setattr(module, "_COMMENT_SUBMIT_READY_TIMEOUT_SECONDS", 0.0)
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": "absent",
                "cleanup": True,
                "dispose": True,
            }
        )
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == "submit_not_ready"
        assert result["retry_safe"] is True
        assert "cleanup" in owner.ran
        assert "submit" not in owner.ran

    async def test_a_control_that_appears_late_is_waited_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wait spans both ``absent`` and ``disabled`` before ``ready``."""
        answers = iter(["absent", "absent", "disabled", "ready"])
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": lambda: next(answers),
                "submit": "clicked",
                "confirm": "confirmed",
                "dispose": True,
            }
        )
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == "posted"
        assert owner.ran.count("submit_ready") == 4
        assert "cleanup" not in owner.ran

    async def test_more_than_one_new_control_is_never_clicked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ambiguous`` is not waited on, because it is not a slow page.

        Several controls appearing at once is a re-render. The reply-suggestion
        chips under a post ("Congratulations!", a bare emoji) are controls too,
        so clicking into that guess posts a canned reply instead of the comment.
        """
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": "ambiguous",
                "cleanup": True,
                "dispose": True,
            }
        )
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == "submit_not_ready"
        assert result["retry_safe"] is True
        assert "submit" not in owner.ran
        # Asked once and not polled: a moving page is not a slow one.
        assert owner.ran.count("submit_ready") == 1

    async def test_a_click_that_did_not_happen_clears_the_text_it_wrote(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": "ready",
                "submit": "invalid",
                "cleanup": True,
                "dispose": True,
            }
        )
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == "submit_failed"
        assert result["retry_safe"] is True
        assert "cleanup" in owner.ran


class TestSubmitted:
    """Past the click, where a retry can comment twice."""

    async def test_a_confirmed_comment_reports_posted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": "ready",
                "submit": "clicked",
                "confirm": "confirmed",
                "dispose": True,
            }
        )
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert result["status"] == "posted"
        assert result["posted"] is True
        assert result["retry_safe"] is False
        # Nothing is cleared after a submit: the text is on the post.
        assert "cleanup" not in owner.ran

    async def test_an_unconfirmed_comment_is_not_safe_to_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(module, "_COMMENT_CONFIRM_TIMEOUT_SECONDS", 0.0)
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": "ready",
                "submit": "clicked",
                "confirm": "pending",
                "dispose": True,
            }
        )
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        # The click happened, so this is not "nothing was submitted": it is an
        # outcome nobody observed, and the flag says so.
        assert result["status"] == "outcome_unknown"
        assert result["posted"] is False
        assert result["retry_safe"] is False
        assert "cleanup" not in owner.ran

    async def test_the_pin_is_released_whatever_the_ending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        owner = _Handle(
            {
                "write": "written",
                "submit_ready": "ready",
                "submit": "clicked",
                "confirm": "confirmed",
                "dispose": True,
            }
        )
        actions, _ = _actions(monkeypatch, surface="ready", owner=owner)

        await actions.comment_on_post(
            ACTIVITY_URN, "Nice write-up.", confirm_comment=True
        )

        assert "dispose" in owner.ran
        assert owner.ran[-1] == "dispose_handle"
