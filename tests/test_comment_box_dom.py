# tests/test_comment_box_dom.py
"""Browser-DOM tests for comment_on_post against a real Chromium DOM.

The unit suite mocks ``page.evaluate``, so none of ``post_actions``' JavaScript
runs there. These tests execute the production programs in headless Chromium
without making a LinkedIn request or write.

What the fixture claims, and what it does not: the shape here was *measured* on
a live post page on 2026-10-27, and the three facts the production code now
depends on are all from that measurement --

1. the page carries **no** ``<form>`` and **no** ``<article>`` inside ``<main>``;
   the post is a ``div[role="listitem"]``,
2. the control that posts a comment is a ``<button type="button">`` with no
   ``aria-label``, reading "Comment" in English only, and
3. it **does not exist** until the editor holds text -- the card's visible
   control count went 14 -> 15 on a real keystroke, and nothing disappeared.

The reply-suggestion chips are reproduced too, as ``div[role="button"]``,
because they are what a looser rule would click instead.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from patchright.async_api import async_playwright

from linkedin_mcp_server.linkedin import post_actions as module
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.post_actions import PostActions
from linkedin_mcp_server.linkedin.session import PageSession


pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

ACTIVITY_URN = "urn:li:activity:7123456789012345678"
REFERENCE = f"/feed/update/{ACTIVITY_URN}/"
POST_URL = f"https://www.linkedin.com{REFERENCE}"

#: The sidebar and the comments the excerpt must not read as the post's words.
#: Both are real: the live page carried the viewer's own profile card and a
#: Premium upsell above the post, and commenters' text below it.
SIDEBAR_TEXT = "Viewer Name Cybersecurity Student Get 4x more profile views"
COMMENTER_TEXT = "Congratulations on the new role"
POST_TEXT = "I'm excited to start my new job as the Director of Marketing"


def _page_html(*, chips: bool = True, extra_controls_on_input: int = 1) -> str:
    """A post page in the measured shape: a list-item card, no form, no article.

    *extra_controls_on_input* is how many controls appear once the editor holds
    text. One is the real page. Zero and two are the failure shapes.
    """
    chip_html = (
        """
        <div role="button" tabindex="0">Congratulations! &#127881;</div>
        <div role="button" tabindex="0">Wishing you the best</div>
        """
        if chips
        else ""
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body>
  <aside><div>{SIDEBAR_TEXT}</div></aside>
  <main>
    <div id="paged-comments">
      <div role="listitem" id="card">
        <button type="button" aria-label="Open control menu for post">&hellip;</button>
        <div id="body">{POST_TEXT}</div>
        <button type="button" aria-label="Reaction button state: no reaction">59</button>
        <button type="button" aria-label="Comment">22</button>
        <button type="button" aria-label="Repost">1</button>
        <div id="composer">
          {chip_html}
          <div role="textbox" contenteditable="true" id="editor"
               aria-label="Text editor for creating comment"></div>
          <button type="button" aria-label="Show Emoji Picker">&#128512;</button>
          <div id="actions"></div>
        </div>
      </div>
      <div id="comments"><div>{COMMENTER_TEXT}</div></div>
    </div>
  </main>
  <script>
    // The control is created from the editor's own input event, which is the
    // behaviour that made the original button[type=submit] rule fail.
    const editor = document.getElementById('editor');
    const actions = document.getElementById('actions');
    let built = false;
    editor.addEventListener('input', () => {{
      const empty = !(editor.innerText || '').trim();
      if (empty || built) return;
      built = true;
      for (let i = 0; i < {extra_controls_on_input}; i += 1) {{
        const button = document.createElement('button');
        button.type = 'button';           // not submit, as measured
        button.textContent = i === 0 ? 'Comment' : 'Something else';
        button.addEventListener('click', () => {{
          document.body.dataset.clicked = 'true';
          // A real submit clears the box and renders the comment under the post.
          document.getElementById('comments').insertAdjacentHTML(
            'beforeend', '<div>' + (editor.innerText || '') + '</div>'
          );
          editor.textContent = '';
        }});
        actions.appendChild(button);
      }}
    }});
  </script>
</body></html>"""


@pytest.fixture
async def dom_page():
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(
                channel="chromium", headless=True
            )
            page = await browser.new_page()
        except Exception as exc:
            if os.environ.get("CI"):
                raise
            pytest.skip(f"chromium unavailable: {exc}")
        page.set_default_timeout(600)
        page.set_default_navigation_timeout(10_000)
        await page.route(
            "https://www.linkedin.com/**",
            lambda route: route.fulfill(
                status=200,
                content_type="text/html",
                body='<!DOCTYPE html><html><head><meta charset="utf-8"></head></html>',
            ),
        )
        try:
            yield page
        finally:
            await browser.close()


async def _load(page, html: str, monkeypatch=None) -> PostActions:
    """Serve *html* at the post's own address, so the route guard is satisfied.

    The navigation and the rate-limit read are stood down when a monkeypatch is
    given: both would replace the page ``set_content`` just put up, and neither
    is what these tests are about. The route guard still runs, against the real
    address the page is sitting on.
    """
    await page.goto(POST_URL)
    await page.set_content(html)
    if monkeypatch is not None:
        monkeypatch.setattr(PageNavigator, "_navigate_to_page", AsyncMock())
        monkeypatch.setattr(PageSession, "check_rate_limit", AsyncMock())
    session = PageSession(page)
    return PostActions(session, PageNavigator(session))


def _argument(comment: str | None = None) -> dict:
    argument: dict = {"reference": REFERENCE}
    if comment is not None:
        argument["comment"] = comment
    return argument


class TestSurface:
    """What the probe answers on a page with no submit button anywhere."""

    async def test_a_post_with_no_submit_button_is_still_ready(self, dom_page):
        # The regression this file exists for: the old rule wanted a
        # button[type="submit"] inside a form or article before typing, and a
        # real post page has none of the three, so every post was refused.
        await _load(dom_page, _page_html())

        state = await dom_page.evaluate(module._COMMENT_SURFACE_JS, _argument())

        assert state == "ready"
        assert (
            await dom_page.evaluate(
                "() => document.querySelectorAll("
                "'main form, main article, main button[type=\"submit\"]').length"
            )
            == 0
        )

    async def test_text_already_in_the_box_is_reported_occupied(self, dom_page):
        await _load(dom_page, _page_html())
        await dom_page.fill("#editor", "half-written thought")

        assert (
            await dom_page.evaluate(module._COMMENT_SURFACE_JS, _argument())
            == "occupied"
        )

    async def test_a_box_outside_any_card_is_refused(self, dom_page):
        await _load(
            dom_page,
            "<!DOCTYPE html><html><body><main>"
            '<div role="textbox" contenteditable="true" id="editor"></div>'
            "</main></body></html>",
        )

        assert (
            await dom_page.evaluate(module._COMMENT_SURFACE_JS, _argument())
            == "no_card"
        )


class TestExcerpt:
    """The excerpt has to be the post's words and nothing else's."""

    async def test_it_reads_the_card_not_the_page(self, dom_page):
        await _load(dom_page, _page_html())

        excerpt = await dom_page.evaluate(
            module._COMMENT_EXCERPT_JS, {**_argument(), "limit": 600}
        )

        assert POST_TEXT in excerpt["text"]
        # The two false positives this scope exists to prevent. A mention check
        # satisfied by either would comment on a post that never said it.
        assert "Get 4x more profile views" not in excerpt["text"]
        assert COMMENTER_TEXT not in excerpt["text"]

    async def test_truncation_is_reported(self, dom_page):
        await _load(dom_page, _page_html())

        excerpt = await dom_page.evaluate(
            module._COMMENT_EXCERPT_JS, {**_argument(), "limit": 12}
        )

        assert len(excerpt["text"]) == 12
        assert excerpt["truncated"] is True

    async def test_another_post_answers_nothing(self, dom_page):
        await _load(dom_page, _page_html())

        excerpt = await dom_page.evaluate(
            module._COMMENT_EXCERPT_JS,
            {
                "reference": "/feed/update/urn:li:activity:7000000000000000009/",
                "limit": 600,
            },
        )

        assert excerpt is None


class TestTheControlThatAppears:
    """Finding the submit control by what was not there before the text was."""

    async def test_the_appearing_control_is_found_and_clicked(
        self, dom_page, monkeypatch
    ):
        actions = await _load(dom_page, _page_html(), monkeypatch)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Thanks for sharing this.", confirm_comment=True
        )

        assert result["status"] == "posted"
        assert result["posted"] is True
        assert result["retry_safe"] is False
        assert await dom_page.evaluate("() => document.body.dataset.clicked") == "true"

    async def test_a_reply_suggestion_chip_is_never_the_control(
        self, dom_page, monkeypatch
    ):
        """The chips are visible controls in the card before the text lands.

        They are therefore in the before-write set and cannot be mistaken for
        the control that appeared -- which is the whole point of diffing nodes
        rather than reading the word on a button.
        """
        actions = await _load(dom_page, _page_html(chips=True), monkeypatch)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Thanks for sharing this.", confirm_comment=True
        )

        assert result["status"] == "posted"
        # The chip's own text never reached the comments list.
        rendered = await dom_page.evaluate(
            "() => document.getElementById('comments').innerText"
        )
        assert "Thanks for sharing this." in rendered
        assert "Wishing you the best" not in rendered.replace(COMMENTER_TEXT, "")

    async def test_no_control_appearing_submits_nothing_and_clears_up(
        self, dom_page, monkeypatch
    ):
        monkeypatch.setattr(module, "_COMMENT_SUBMIT_READY_TIMEOUT_SECONDS", 0.3)
        actions = await _load(
            dom_page, _page_html(extra_controls_on_input=0), monkeypatch
        )

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Thanks for sharing this.", confirm_comment=True
        )

        assert result["status"] == "submit_not_ready"
        assert result["retry_safe"] is True
        assert await dom_page.evaluate("() => document.body.dataset.clicked") is None
        # The text this call wrote does not outlive it.
        assert (
            await dom_page.evaluate(
                "() => document.getElementById('editor').innerText.trim()"
            )
            == ""
        )

    async def test_two_new_controls_are_refused_rather_than_guessed(
        self, dom_page, monkeypatch
    ):
        monkeypatch.setattr(module, "_COMMENT_SUBMIT_READY_TIMEOUT_SECONDS", 0.3)
        actions = await _load(
            dom_page, _page_html(extra_controls_on_input=2), monkeypatch
        )

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Thanks for sharing this.", confirm_comment=True
        )

        assert result["status"] == "submit_not_ready"
        assert result["retry_safe"] is True
        assert await dom_page.evaluate("() => document.body.dataset.clicked") is None

    async def test_a_dry_run_types_nothing_and_builds_no_control(
        self, dom_page, monkeypatch
    ):
        actions = await _load(dom_page, _page_html(), monkeypatch)

        result = await actions.comment_on_post(
            ACTIVITY_URN, "Thanks for sharing this.", confirm_comment=False
        )

        assert result["status"] == "dry_run"
        assert POST_TEXT in result["excerpt"]
        assert (
            await dom_page.evaluate(
                "() => document.getElementById('editor').innerText.trim()"
            )
            == ""
        )
        assert (
            await dom_page.evaluate(
                "() => document.getElementById('actions').children.length"
            )
            == 0
        )
