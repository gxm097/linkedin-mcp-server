"""Browser-UI comment composition and submit workflow for one post.

``posts.py`` searches content and stays a reader; this module is the only one
that types into a post. It follows ``message_sender.py`` rather than inventing
a second write model, because the hazard is the same one: a contenteditable
surface, a submit button that enables itself a frame later, and a page that may
replace either node between two awaits. The answer is the same too -- the
editor and its button are pinned once on a node held across the whole call, and
every step revalidates that pin before it acts.

Per the AGENTS.md LinkedIn Page Rules nothing here reads a label value. The
post being commented on is established by the *route* -- the document's own
address has to be the permalink the caller named -- and never by matching text
or a class name inside the page. That choice is what makes the identity check
measurable: ``/feed/update/<urn>/`` carries the entity id in the URL, so a
comment can only land on the post whose permalink the browser is sitting on.
A post page also has exactly one comment box of its own, so the editor is
accepted only while it is the single visible candidate inside ``<main>``; a
reply box under an existing comment opens only on a click this module never
makes, and the messaging overlay lives outside ``<main>``.

The write gate is ``confirm_comment``. False probes the surface and returns
without touching the editor, which is the only way to learn whether a post
accepts comments at all -- LinkedIn offers no read that answers it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import anyio
import anyio.lowlevel

import linkedin_mcp_server.linkedin.contracts as contracts
from linkedin_mcp_server.linkedin.identifiers import (
    normalize_post_reference,
    post_permalink_url,
)
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.session import PageSession

logger = logging.getLogger(__name__)

# The same shape message_sender pins: LinkedIn builds both composers on the
# same rich-text editor, so the selector is shared rather than re-derived.
_COMMENT_EDITOR_SELECTOR = '[role="textbox"][contenteditable="true"]'

# How long the comment box is given to appear after the permalink loads. A post
# page renders its own box with the post, but the editor is hydrated client-side
# and a cold browser reaches it a beat after the text is readable.
_COMMENT_SURFACE_TIMEOUT_SECONDS = 8.0
# How long the submit button is given to leave its disabled state once the text
# is in. LinkedIn enables it from an input event, so this is frames, not pages.
_COMMENT_SUBMIT_READY_TIMEOUT_SECONDS = 5.0
# How long the posted comment is given to appear under the post.
_COMMENT_CONFIRM_TIMEOUT_SECONDS = 12.0
_COMMENT_CLEANUP_TIMEOUT_SECONDS = 5.0
_SURFACE_POLL_SECONDS = 0.15

# How much of the post's own text a dry run hands back. Enough to tell one post
# from another, which is all the excerpt is for; a probe is not a content read,
# and the readers in posts.py and feed.py are what fetch a post in full.
_POST_EXCERPT_CHARACTERS = 600

# Shared page-side helpers. Inlined into each script rather than installed on
# the page once: a script that depends on a previous evaluation having run is a
# script that fails open when the document was replaced between the two.
_COMMENT_SHARED_JS = r"""
    const visible = element => {
        const visibility = element && getComputedStyle(element).visibility;
        return !!(
            element &&
            visibility !== 'hidden' &&
            visibility !== 'collapse' &&
            (element.offsetWidth || element.offsetHeight || element.getClientRects().length)
        );
    };
    // The document's own address has to be the permalink the caller named.
    // Compared as a path, because LinkedIn answers the same post on a locale
    // subdomain and appends tracking query parameters of its own.
    const onPostRoute = reference => {
        try {
            const url = new URL(window.location.href);
            const hostname = url.hostname.toLowerCase().replace(/\.$/, '');
            if (
                url.protocol !== 'https:' ||
                !/(^|\.)linkedin\.com$/.test(hostname) ||
                url.username ||
                url.password ||
                (url.port && url.port !== '443')
            ) {
                return false;
            }
            const path = decodeURIComponent(url.pathname);
            const target = decodeURIComponent(reference);
            // A trailing slash is the same post on both routes.
            return path.replace(/\/$/, '') === target.replace(/\/$/, '');
        } catch {
            return false;
        }
    };
    // Only the post's own box. The messaging overlay sits outside <main>, and a
    // reply box under an existing comment never opens without a click.
    const commentEditors = () => {
        const main = document.querySelector('main');
        if (!main) return [];
        return Array.from(
            main.querySelectorAll('[role="textbox"][contenteditable="true"]')
        ).filter(visible);
    };
    const soleEditor = () => {
        const editors = commentEditors();
        return editors.length === 1 ? editors[0] : null;
    };
    // The post's own card: the innermost container that holds this comment box
    // and nothing of any neighbouring post. Measured on a live post page, the
    // page carries no <form> and no <article> anywhere inside <main> -- the
    // card is a list item. An ARIA role is structure, not a class name and not
    // a label value, so it is what the pin hangs on. A form or an article is
    // still preferred where one exists, which is the shape messaging uses.
    const cardFor = editor => {
        let ancestor = editor?.parentElement;
        while (ancestor) {
            if (ancestor.matches('form, article, [role="listitem"]')) return ancestor;
            ancestor = ancestor.parentElement;
        }
        return null;
    };
    // Every control the card offers, the editor's own insides excluded. Both
    // tags and roles, because LinkedIn builds some of these as divs, and the
    // reply suggestions under a post are exactly that.
    const cardControls = (card, editor) =>
        Array.from(card.querySelectorAll('button, [role="button"]')).filter(
            candidate =>
                visible(candidate) &&
                candidate !== editor &&
                !editor.contains(candidate) &&
                !candidate.contains(editor)
        );
    const controlDisabled = control =>
        control.disabled === true ||
        (control.getAttribute('aria-disabled') || '').toLowerCase() === 'true';
    const semanticAncestors = element => {
        const scopes = [];
        let ancestor = element?.parentElement;
        while (ancestor) {
            if (
                ancestor.matches(
                    'form, article, dialog, [role="dialog"], main, [role="listitem"]'
                )
            ) {
                scopes.push(ancestor);
            }
            ancestor = ancestor.parentElement;
        }
        return scopes;
    };
    // Line-exact comparison of what the editor holds. LinkedIn's editor wraps
    // each line in a bare <p>, and a non-breaking space is the same character
    // to a reader, so both are normalized before comparing.
    const editorLines = editor => {
        const paragraphs = Array.from(editor.childNodes);
        if (paragraphs.length === 0) return null;
        const lines = [];
        for (const paragraph of paragraphs) {
            if (
                paragraph.nodeType !== Node.ELEMENT_NODE ||
                paragraph.tagName !== 'P' ||
                paragraph.attributes.length !== 0
            ) {
                return null;
            }
            const children = Array.from(paragraph.childNodes);
            let line = '';
            for (const [index, child] of children.entries()) {
                if (child.nodeType === Node.TEXT_NODE) {
                    line += child.data;
                } else if (
                    child.nodeType !== Node.ELEMENT_NODE ||
                    child.tagName !== 'BR' ||
                    index !== children.length - 1
                ) {
                    return null;
                }
            }
            lines.push(line);
        }
        return lines;
    };
    const sameLines = (lines, expected) =>
        lines !== null &&
        lines.length === expected.length &&
        lines.every(
            (line, index) =>
                line.replace(/ /g, ' ') === expected[index].replace(/ /g, ' ')
        );
    const editorMatches = (editor, text) => text.includes('\n')
        ? sameLines(editorLines(editor), text.split('\n'))
        : (editor.innerText || editor.textContent || '') === text;
    const editorEmpty = editor =>
        !(editor.innerText || '').replace(/\s+/g, ' ').trim();
"""

# Revalidates the pin before every act. Mirrors message_sender's validatePinned:
# the pinned nodes must still be connected, still each other's relatives in the
# same order, still visible, and the route must still be the target post.
_COMMENT_PINNED_JS = (
    _COMMENT_SHARED_JS
    + r"""
    const validatePinned = (arg, requireEnabled = false) => {
        const pinned = owner?.__linkedinMcpComment;
        if (!pinned || pinned.reference !== arg.reference) return null;
        if (!onPostRoute(arg.reference)) return null;
        const {editor, button, ancestorChain} = pinned;
        const currentChain = semanticAncestors(editor);
        if (
            !owner.isConnected ||
            !editor?.isConnected ||
            !Array.isArray(ancestorChain) ||
            currentChain.length !== ancestorChain.length ||
            currentChain.some((scope, index) => scope !== ancestorChain[index]) ||
            !currentChain.includes(owner) ||
            !owner.contains(editor) ||
            !visible(editor) ||
            !editor.matches('[role="textbox"][contenteditable="true"]')
        ) {
            return null;
        }
        // Still the only comment box on the page, and still this one. A second
        // box means a reply surface opened under us and the pin is ambiguous.
        if (soleEditor() !== editor) return null;
        // The post's comment control does not exist until the editor holds
        // text, so before the write the pin carries no button and null is the
        // valid state. Once found it is held to everything the editor is, and
        // a step that means to click still refuses a pin without one.
        if (button === null) {
            if (requireEnabled) return null;
        } else if (
            !button.isConnected ||
            !owner.contains(button) ||
            !visible(button) ||
            editor.contains(button) ||
            button === editor ||
            (requireEnabled && controlDisabled(button))
        ) {
            return null;
        }
        return pinned;
    };
"""
)

# Probe only. Answers whether this post offers a comment box, without focusing
# or typing, so the dry run is as read-only as any other page read.
_COMMENT_SURFACE_JS = (
    "(arg) => {"
    + _COMMENT_SHARED_JS
    + r"""
    if (!onPostRoute(arg.reference)) return 'off_route';
    const editors = commentEditors();
    if (editors.length === 0) return 'absent';
    if (editors.length > 1) return 'ambiguous';
    const editor = editors[0];
    // The card only. The submit control is deliberately not required here:
    // on a post page it does not exist until the editor holds text, so
    // demanding it before typing refuses every post that accepts comments.
    if (!cardFor(editor)) return 'no_card';
    return editorEmpty(editor) ? 'ready' : 'occupied';
    }"""
)

# Reads the post's own words, so a caller can tell this post from another one.
# Route-guarded like every other script here: text off the target post is worse
# than no text, because it would be matched against and believed.
_COMMENT_EXCERPT_JS = (
    "(arg) => {"
    + _COMMENT_SHARED_JS
    + r"""
    if (!onPostRoute(arg.reference)) return null;
    const main = document.querySelector('main');
    if (!main) return null;
    // The post's own card, never the whole of <main>. Measured on a live post
    // page: <main> carried the viewer's own sidebar profile, a Premium upsell
    // and every commenter's words -- 1679 characters against the card's 462 --
    // so a mention check against it answers yes on words the post never said.
    // Anchored on the comment box, because that box belongs to this post and
    // the first card in the document order need not.
    const editor = soleEditor();
    const scope =
        (editor && cardFor(editor)) ||
        Array.from(main.querySelectorAll('[role="listitem"], article')).filter(
            visible
        )[0] ||
        main;
    const text = (scope.innerText || '')
        .replace(/[^\S\n]+/g, ' ')
        .replace(/ ?\n ?/g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
    if (!text) return null;
    return {text: text.slice(0, arg.limit), truncated: text.length > arg.limit};
    }"""
)

# Pins the editor and its ancestor chain onto the post's card and hands that
# node back, so every later step evaluates against the same element rather than
# re-querying a page that may have moved on. The submit control is not pinned
# here because it does not exist yet; what is recorded instead is the set of
# controls the card offered *before* anything was typed, which is what later
# identifies the one that appears.
_COMMENT_OWNER_JS = (
    "(arg) => {"
    + _COMMENT_SHARED_JS
    + r"""
    if (!onPostRoute(arg.reference)) return null;
    const editor = soleEditor();
    if (!editor) return null;
    const card = cardFor(editor);
    if (!card) return null;
    const ancestorChain = semanticAncestors(editor);
    if (!ancestorChain.includes(card)) return null;
    card.__linkedinMcpComment = {
        editor,
        button: null,
        // Nodes, not labels. The control that posts a comment carries no
        // aria-label and is told apart from the reply-suggestion chips beside
        // it only by not having been here before the text was -- a structural
        // count, which is what the locale rule asks for.
        controlsBeforeWrite: cardControls(card, editor),
        ancestorChain,
        reference: arg.reference,
        ownedComment: null,
    };
    return card;
    }"""
)

_COMMENT_WRITE_JS = (
    "(owner, arg) => {"
    + _COMMENT_PINNED_JS
    + r"""
    let pinned = validatePinned(arg);
    if (!pinned) return 'invalid';
    const {editor} = pinned;
    // Never overwrite what someone else typed. A half-written comment in the
    // box is the user's, and this call is not authorized to replace it.
    if (!editorEmpty(editor)) return 'occupied';
    editor.focus();
    pinned = validatePinned(arg);
    if (!pinned || document.activeElement !== editor) return 'invalid';
    if (!editorEmpty(editor)) return 'occupied';
    // insertText is what makes the editor's own input handlers run, which is
    // what enables the submit button. Assigning innerText does neither.
    if (
        typeof document.queryCommandSupported !== 'function' ||
        !document.queryCommandSupported('insertText') ||
        typeof document.execCommand !== 'function'
    ) {
        return 'unsupported';
    }
    const inserted = document.execCommand('insertText', false, arg.comment);
    if (editorMatches(editor, arg.comment)) {
        pinned.ownedComment = arg.comment;
    }
    if (inserted !== true) return 'unsupported';
    pinned = validatePinned(arg);
    if (
        !pinned ||
        document.activeElement !== editor ||
        pinned.ownedComment !== arg.comment ||
        !editorMatches(editor, arg.comment)
    ) {
        return 'invalid';
    }
    return 'written';
    }"""
)

# Finds the control that posts the comment, then reports whether it can be
# clicked. Identity is the one thing measured here: the control that was not in
# the card before the text was written. Never the word on it -- the button
# carries no aria-label and reads "Comment" only in English, and the chips
# sitting beside it ("Congratulations!", a bare emoji) are controls too, so a
# rule that guessed would post a canned reply instead of the comment.
_COMMENT_SUBMIT_READY_JS = (
    "(owner, arg) => {"
    + _COMMENT_PINNED_JS
    + r"""
    const pinned = validatePinned(arg);
    if (
        !pinned ||
        pinned.ownedComment !== arg.comment ||
        !editorMatches(pinned.editor, arg.comment)
    ) {
        return 'invalid';
    }
    if (pinned.button === null) {
        if (!Array.isArray(pinned.controlsBeforeWrite)) return 'invalid';
        const appeared = cardControls(owner, pinned.editor).filter(
            control => !pinned.controlsBeforeWrite.includes(control)
        );
        // Nothing yet: the control is hydrated a frame or two after the input
        // event, so an empty answer is worth waiting on.
        if (appeared.length === 0) return 'absent';
        // More than one is a re-render, not a submit button, and this call
        // does not guess which of them posts a comment.
        if (appeared.length > 1) return 'ambiguous';
        pinned.button = appeared[0];
        // Hold the newly pinned control to every check the editor just passed.
        if (!validatePinned(arg)) {
            pinned.button = null;
            return 'invalid';
        }
    }
    return controlDisabled(pinned.button) ? 'disabled' : 'ready';
    }"""
)

_COMMENT_SUBMIT_JS = (
    "(owner, arg) => {"
    + _COMMENT_PINNED_JS
    + r"""
    const pinned = validatePinned(arg, true);
    if (
        !pinned ||
        pinned.ownedComment !== arg.comment ||
        !editorMatches(pinned.editor, arg.comment)
    ) {
        return 'invalid';
    }
    pinned.submitted = true;
    pinned.button.click();
    return 'clicked';
    }"""
)

# Confirmation is two observations at once: our text left the editor, and it is
# now rendered somewhere under the post outside the editor. Either alone is a
# false positive -- an editor clears on a failed submit too, and the text could
# be read back out of the box that still holds it.
_COMMENT_CONFIRM_JS = (
    "(owner, arg) => {"
    + _COMMENT_SHARED_JS
    + r"""
    const pinned = owner?.__linkedinMcpComment;
    if (!pinned || pinned.reference !== arg.reference) return 'invalid';
    if (!onPostRoute(arg.reference)) return 'invalid';
    const {editor} = pinned;
    if (!editor?.isConnected) return 'invalid';
    if (!editorEmpty(editor)) return 'pending';
    const main = document.querySelector('main');
    if (!main) return 'invalid';
    const expected = arg.comment.replace(/\s+/g, ' ').trim();
    const rendered = Array.from(main.querySelectorAll('article, li, div')).some(
        node =>
            !editor.contains(node) &&
            node !== editor &&
            !node.contains(editor) &&
            visible(node) &&
            (node.innerText || '').replace(/\s+/g, ' ').trim().includes(expected)
    );
    return rendered ? 'confirmed' : 'pending';
    }"""
)

# Only ever removes text this call proved it put there and never submitted.
_COMMENT_CLEANUP_JS = (
    "(owner, arg) => {"
    + _COMMENT_SHARED_JS
    + r"""
    const pinned = owner?.__linkedinMcpComment;
    if (
        !pinned ||
        pinned.submitted ||
        pinned.ownedComment !== arg.comment ||
        !pinned.editor?.isConnected ||
        !editorMatches(pinned.editor, arg.comment)
    ) {
        return false;
    }
    const {editor} = pinned;
    editor.focus();
    if (document.activeElement !== editor) return false;
    const range = document.createRange();
    range.selectNodeContents(editor);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    if (
        typeof document.queryCommandSupported === 'function' &&
        document.queryCommandSupported('insertText') &&
        typeof document.execCommand === 'function'
    ) {
        document.execCommand('insertText', false, '');
    }
    pinned.ownedComment = null;
    return editorEmpty(editor);
    }"""
)

_COMMENT_DISPOSE_JS = r"""(owner) => {
    if (owner && owner.__linkedinMcpComment) {
        delete owner.__linkedinMcpComment;
    }
    return true;
}"""


class PostActions:
    """Compose and submit one comment on one LinkedIn post."""

    def __init__(self, session: PageSession, navigator: PageNavigator):
        self._session = session
        self._page = session.page
        self._navigator = navigator

    @staticmethod
    def _argument(reference: str, comment: str | None = None) -> dict[str, Any]:
        argument: dict[str, Any] = {"reference": reference}
        if comment is not None:
            argument["comment"] = comment
        return argument

    async def _wait_for_comment_surface(self, reference: str) -> str:
        """Poll for the post's own comment box, and answer what it found."""
        deadline = self._session.monotonic() + _COMMENT_SURFACE_TIMEOUT_SECONDS
        state = "absent"
        while True:
            state = await self._session.run_on_linkedin(
                _COMMENT_SURFACE_JS, self._argument(reference)
            )
            if state in {"ready", "occupied", "off_route"}:
                return str(state)
            remaining = deadline - self._session.monotonic()
            if remaining <= 0:
                return str(state)
            await self._session.delay(min(_SURFACE_POLL_SECONDS, remaining))

    async def _post_excerpt(self, reference: str) -> tuple[str | None, bool]:
        """Read the post's own words, and say whether they were cut short.

        Answers ``(None, False)`` when the text could not be read at all. A
        failure here never fails the call: the excerpt tells a caller which
        post it reached, and losing that is worth strictly less than losing
        the surface verdict the probe went there for.
        """
        try:
            excerpt = await self._session.run_on_linkedin(
                _COMMENT_EXCERPT_JS,
                {**self._argument(reference), "limit": _POST_EXCERPT_CHARACTERS},
            )
        except Exception:
            logger.debug("Could not read the post's text", exc_info=True)
            return None, False
        if not isinstance(excerpt, dict):
            return None, False
        text = excerpt.get("text")
        if not isinstance(text, str) or not text:
            return None, False
        return text, bool(excerpt.get("truncated"))

    async def _resolve_owner(self, reference: str) -> Any | None:
        """Hold the pinned scope node across submission and confirmation."""
        owner = await self._page.evaluate_handle(
            _COMMENT_OWNER_JS, self._argument(reference)
        )
        if owner.as_element() is None:
            await self._dispose_owner(owner)
            return None
        return owner

    @staticmethod
    async def _dispose_owner(owner: Any) -> None:
        """Release the pin and the handle, whatever the call did."""
        try:
            with contracts.before_the_reply_deadline(
                _COMMENT_CLEANUP_TIMEOUT_SECONDS, shield=True
            ) as dom_scope:
                try:
                    await owner.evaluate(_COMMENT_DISPOSE_JS)
                except Exception:
                    logger.debug(
                        "Could not clear the pinned comment nodes", exc_info=True
                    )
            if dom_scope.cancel_called:
                logger.warning("Timed out clearing the pinned comment nodes")
        finally:
            with contracts.before_the_reply_deadline(
                _COMMENT_CLEANUP_TIMEOUT_SECONDS, shield=True
            ) as handle_scope:
                try:
                    await owner.dispose()
                except Exception:
                    logger.debug(
                        "Could not release the comment owner handle", exc_info=True
                    )
            if handle_scope.cancel_called:
                logger.warning("Timed out releasing the comment owner handle")
        await anyio.lowlevel.checkpoint()

    async def _write(self, comment: str, *, reference: str, owner: Any) -> str:
        """Insert the comment into the pinned editor."""
        result = await owner.evaluate(
            _COMMENT_WRITE_JS, self._argument(reference, comment)
        )
        return result if result in {"written", "occupied", "unsupported"} else "invalid"

    async def _wait_for_submit(
        self, comment: str, *, reference: str, owner: Any
    ) -> bool:
        """Wait for the comment's own control to appear and become clickable.

        ``absent`` and ``disabled`` are both worth waiting on: the control is
        created from the editor's input event and hydrated a frame or two
        later, so it is first missing and then briefly inert. ``ambiguous``
        is not waited on at all -- more than one control appearing means the
        card re-rendered, and the next poll would be guessing against a moving
        page rather than reading a slower one.
        """
        deadline = time.monotonic() + _COMMENT_SUBMIT_READY_TIMEOUT_SECONDS
        argument = self._argument(reference, comment)
        while True:
            try:
                state = await owner.evaluate(_COMMENT_SUBMIT_READY_JS, argument)
            except Exception:
                logger.debug("Could not read the comment's control", exc_info=True)
                return False
            if state == "ready":
                return True
            if state not in {"disabled", "absent"}:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.05, remaining))

    async def _submit(self, comment: str, *, reference: str, owner: Any) -> str:
        """Click the one pinned, enabled submit button."""
        result = await owner.evaluate(
            _COMMENT_SUBMIT_JS, self._argument(reference, comment)
        )
        return "clicked" if result == "clicked" else "invalid"

    async def _confirmed(self, comment: str, *, reference: str, owner: Any) -> bool:
        """Wait for the comment to appear on the post under this account."""
        deadline = time.monotonic() + _COMMENT_CONFIRM_TIMEOUT_SECONDS
        argument = self._argument(reference, comment)
        while True:
            try:
                state = await owner.evaluate(_COMMENT_CONFIRM_JS, argument)
            except Exception:
                logger.debug("Could not confirm the comment", exc_info=True)
                return False
            if state == "confirmed":
                return True
            if state == "invalid":
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_SURFACE_POLL_SECONDS, remaining))

    async def _cleanup(self, comment: str, *, reference: str, owner: Any) -> None:
        """Best-effort removal of text proven to belong to this call."""
        with contracts.before_the_reply_deadline(
            _COMMENT_CLEANUP_TIMEOUT_SECONDS, shield=True
        ) as scope:
            try:
                await owner.evaluate(
                    _COMMENT_CLEANUP_JS, self._argument(reference, comment)
                )
            except Exception:
                logger.debug("Could not clear tool-owned comment text", exc_info=True)
        if scope.cancel_called:
            logger.warning("Timed out clearing tool-owned comment text")

    async def comment_on_post(
        self,
        post: str,
        comment: str,
        *,
        confirm_comment: bool = False,
    ) -> dict[str, Any]:
        """Comment on *post*, or report what a comment would meet.

        Raises:
            InvalidReferenceError: when *post* names no post.
        """
        reference = normalize_post_reference(post)
        url = post_permalink_url(reference)
        refusal = contracts.refuse_an_invalid_comment(url, comment)
        if refusal is not None:
            return refusal
        # The text LinkedIn would keep, so what is compared against the editor
        # and against the posted comment is what actually gets submitted.
        text = contracts.normalize_message_text(comment)

        result = contracts.comment_action_result
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()

        surface = await self._wait_for_comment_surface(reference)
        if surface == "off_route":
            return result(
                url,
                "post_unavailable",
                "That permalink did not stay on the post. It may be deleted, "
                "restricted to an audience this account is not in, or the "
                "reference may name a post that never existed.",
            )
        if surface == "absent":
            return result(
                url,
                "comments_unavailable",
                "The post offers no comment box to this account. The author may "
                "have restricted who can comment.",
            )
        if surface == "ambiguous":
            return result(
                url,
                "comments_unavailable",
                "More than one comment box is open on the page, so the one "
                "belonging to this post cannot be told apart. Reload and retry.",
            )
        if surface == "no_card":
            return result(
                url,
                "comments_unavailable",
                "The comment box could not be tied to the post it belongs to, "
                "so a comment typed into it might land under another post. "
                "Reload and retry.",
            )
        if surface == "occupied":
            return result(
                url,
                "editor_occupied",
                "The comment box already holds text this call did not write. "
                "Clear or post it before commenting through the tool.",
            )

        if not confirm_comment:
            # Read the post only on the probe. The confirmed path is a write
            # and does not need telling which post it is on: a caller that
            # cared has already dry-run it, and the route guard is what keeps
            # the comment on the post either way.
            excerpt, truncated = await self._post_excerpt(reference)
            return result(
                url,
                "dry_run",
                "The post accepts a comment from this account. Nothing was "
                "typed or submitted. Check the excerpt is the post you meant, "
                "then call again with confirm_comment True to post it.",
                excerpt=excerpt,
                excerpt_truncated=truncated,
            )

        owner = await self._resolve_owner(reference)
        if owner is None:
            return result(
                url,
                "comments_unavailable",
                "The comment box could not be pinned to this post. Reload and retry.",
            )
        try:
            written = await self._write(text, reference=reference, owner=owner)
            if written == "occupied":
                return result(
                    url,
                    "editor_occupied",
                    "The comment box already holds text this call did not "
                    "write. Clear or post it before commenting through the tool.",
                )
            if written == "unsupported":
                return result(
                    url,
                    "editor_unsupported",
                    "The comment box did not accept programmatic text entry.",
                )
            if written != "written":
                await self._cleanup(text, reference=reference, owner=owner)
                return result(
                    url,
                    "editor_changed",
                    "The page changed while the comment was being written, so "
                    "nothing was submitted. Retry.",
                )
            if not await self._wait_for_submit(text, reference=reference, owner=owner):
                await self._cleanup(text, reference=reference, owner=owner)
                return result(
                    url,
                    "submit_not_ready",
                    "The control that posts the comment never appeared or "
                    "never became active, so nothing was submitted. The "
                    "comment may be longer than LinkedIn allows.",
                )
            if await self._submit(text, reference=reference, owner=owner) != "clicked":
                await self._cleanup(text, reference=reference, owner=owner)
                return result(
                    url,
                    "submit_failed",
                    "The post button could not be clicked, so nothing was "
                    "submitted. Retry.",
                )
            # Past the click nothing is retry-safe: a comment that posted and
            # failed to confirm reads exactly like one that never left the box.
            confirmed = await self._confirmed(text, reference=reference, owner=owner)
            if confirmed:
                return result(
                    url,
                    "posted",
                    "The comment was submitted and is showing on the post.",
                    posted=True,
                    retry_safe=False,
                )
            return result(
                url,
                "outcome_unknown",
                "The comment was submitted but was not observed on the post. It "
                "may have posted. Open the post and check before retrying; a "
                "retry can comment twice.",
                posted=False,
                retry_safe=False,
            )
        finally:
            await self._dispose_owner(owner)
